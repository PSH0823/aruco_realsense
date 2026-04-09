#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect an ArUco marker with a RealSense camera and publish
<base_frame> → <qr_frame> as a TF in real time.

Key:
    q : quit node
"""

import argparse
from collections import deque
import yaml
import numpy as np
import cv2
import pyrealsense2 as rs
import rospy
import tf2_ros
import tf.transformations as tft
from geometry_msgs.msg import PoseStamped
from math import sqrt, atan2, asin

# import os

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_intrinsics(path):
    with open(path, 'r') as f:
        y = yaml.safe_load(f)
    K   = np.array(y['camera_matrix'], dtype=np.float32)
    dst = np.array(y['dist_coeff'],    dtype=np.float32)
    return K, dst

def load_yaml_head2cam_tf(path):
    with open(path, 'r') as f:
        d = yaml.safe_load(f)
    euler_zyx = d["euler_zyx"]  # Order: roll, pitch, yaw
    trans = d["translation"]
    R = tft.euler_matrix(euler_zyx[0], euler_zyx[1], euler_zyx[2])[:3,:3]
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = np.array(trans)

    return T

def overlay_help(img, text, alpha=0.6):
    """Draw translucent banner with instructions."""
    banner = img.copy()
    h, w = img.shape[:2]
    pad, y0 = 10, 30
    cv2.rectangle(banner, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(banner, text, (pad, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.addWeighted(banner, alpha, img, 1-alpha, 0, img)


# ------------------------------------------------------------
# Simple moving‑average filter
# ------------------------------------------------------------
class PoseFilter:
    def __init__(self, window=10):
        self.buf = deque(maxlen=window)

    def add(self, p, q):
        self.buf.append((p, q))

    def get(self):
        if not self.buf:
            return None
        pos = np.mean([b[0] for b in self.buf], axis=0)
        quat = np.mean([b[1] for b in self.buf], axis=0)
        n = np.linalg.norm(quat)
        if n > 1e-9:
            quat /= n
        return pos, quat


# ------------------------------------------------------------
# Main node
# ------------------------------------------------------------
class QRPublisher:
    def __init__(self, args):
        rospy.init_node('qr_tf_publisher', anonymous=True)

        # TF
        self.tfb   = tf2_ros.TransformBroadcaster()
        self.buf   = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.buf)

        # Publisher
        self.base2cam_pub = rospy.Publisher('/camera/qr_pose', PoseStamped, queue_size=1)                                         # send TF from base to cam

        # Subscriber
        self.base2head_sub = rospy.Subscriber("/tocabi_cc/base_to_head", PoseStamped, self.base2head_callback, queue_size=1)      # receive TF from base to head


        # RealSense
        self.pipe  = rs.pipeline()
        cfg        = rs.config()
        cfg.enable_stream(rs.stream.color, args.width, args.height,
                          rs.format.rgb8, args.fps)
        self.pipe.start(cfg)

        # Intrinsics
        if args.intrinsics:
            self.K, self.dist = load_intrinsics(args.intrinsics)
            rospy.loginfo("Loaded intrinsics from %s", args.intrinsics)
        else:
            intr = self.pipe.get_active_profile() \
                      .get_stream(rs.stream.color) \
                      .as_video_stream_profile() \
                      .get_intrinsics()
            self.K = np.array([[intr.fx, 0, intr.ppx],
                               [0, intr.fy, intr.ppy],
                               [0, 0, 1]], dtype=np.float32)
            self.dist = np.array(intr.coeffs[:5], dtype=np.float32)

            rospy.loginfo("Intrinsics auto-fetched from RealSense")

        # if args.intrinsics and os.path.exists(args.intrinsics):
        #     self.K, self.dist = load_intrinsics(args.intrinsics)
        #     rospy.loginfo("Loaded intrinsics from %s", args.intrinsics)
        # else:
        #     intr = self.pipe.get_active_profile() \
        #               .get_stream(rs.stream.color) \
        #               .as_video_stream_profile() \
        #               .get_intrinsics()
        #     self.K = np.array([[intr.fx, 0, intr.ppx],
        #                        [0, intr.fy, intr.ppy],
        #                        [0, 0, 1]], dtype=np.float32)
        #     self.dist = np.array(intr.coeffs[:5], dtype=np.float32)

        #     rospy.loginfo("Intrinsics auto-fetched from RealSense")

        # ArUco
        self.dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, 'DICT_'+args.dictionary))
        self.par  = cv2.aruco.DetectorParameters()
        self.mlen = args.marker_length

        # 4개 마커 정의
        # 1 0
        # 2 3
        ids = np.array([[1], [0], [2], [3]], dtype=np.int32)

        s = self.mlen           # marker size (예: 0.03 or 0.04)
        d = 0.005               

        corners = [
            # ID 1 (x-, y-)
            np.array([[-d-s, -d-s, 0],
                    [-d,   -d-s, 0],
                    [-d,   -d,   0],
                    [-d-s, -d,   0]], dtype=np.float32),

            # ID 0 (x+, y-)
            np.array([[ d,   -d-s, 0],
                    [ d+s, -d-s, 0],
                    [ d+s, -d,   0],
                    [ d,   -d,   0]], dtype=np.float32),

            # ID 2 (x-, y+)
            np.array([[-d-s, d, 0],
                    [-d,   d, 0],
                    [-d, d+s, 0],
                    [-d-s, d+s, 0]], dtype=np.float32),

            # ID 3 (x+, y+)
            np.array([[ d,  d, 0],
                    [ d+s, d, 0],
                    [ d+s, d+s, 0],
                    [ d,  d+s, 0]], dtype=np.float32)
        ]

        self.board = cv2.aruco.Board(corners, self.dict, ids)

        # TF
        self.base2head = None
        self.head2cam = load_yaml_head2cam_tf("TF_head2cam.yaml")

        # Frames & filter
        self.base  = args.base_frame
        self.head   = args.head_frame
        self.qr    = args.qr_frame
        self.filt  = PoseFilter(args.filter_N)

        # Display
        self.gui = args.display
        if self.gui:
            cv2.namedWindow("Live", cv2.WINDOW_NORMAL)

    # --------------------------------------------------------
    # Run loop
    # --------------------------------------------------------
    def spin(self):
        rate = rospy.Rate(30)   # if you want to run at camera fps
        counter = 0             # for printing TF every 15 frames

        while not rospy.is_shutdown():
            frame    = self.pipe.wait_for_frames().get_color_frame()
            img      = cv2.cvtColor(np.asanyarray(frame.get_data()),
                                    cv2.COLOR_RGB2BGR)

            cam2qr = self.detect_marker(img)

            if self.base2head is not None and cam2qr is not None:
                base2qr = self.base2head @ self.head2cam @ cam2qr
                p = base2qr[:3, 3]
                if (counter % 15 == 0):
                    rospy.loginfo(f"base2qr translation:{p}")
                q = tft.quaternion_from_matrix(base2qr)
                # self.filt.add(p, q)
                # out = self.filt.get()
                out = p, q
                if out:
                    self.publish_tf(out[0], out[1])

            if self.gui:
                overlay_help(img, "q: quit")
                cv2.imshow("Live", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            counter += 1
            rate.sleep()

        self.pipe.stop()
        cv2.destroyAllWindows()

    # --------------------------------------------------------
    # Detect first ArUco marker → cam→qr 4×4
    # --------------------------------------------------------
    def detect_marker(self, img_bgr):
        corners, ids, _ = cv2.aruco.detectMarkers(
            img_bgr, self.dict, parameters=self.par
        )

        if ids is None:
            return None

        # Board 기반 pose 추정
        retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
            corners, ids, self.board, self.K, self.dist, None, None
        )

        if retval <= 0:
            return None

        # 시각화
        cv2.aruco.drawDetectedMarkers(img_bgr, corners, ids)
        cv2.drawFrameAxes(img_bgr, self.K, self.dist,
                        rvec, tvec, self.mlen)

        # 4x4 변환행렬
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()

        return T

    # --------------------------------------------------------
    # Publish TF  base → qr  (dynamic broadcaster)
    # --------------------------------------------------------
    def publish_tf(self, pos, quat):
        ps = PoseStamped()
        ps.header.stamp = rospy.Time.now()
        ps.header.frame_id = self.base

        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = pos
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = quat
        self.base2cam_pub.publish(ps)

    # --------------------------------------------------------
    # Get TF from base frame to head frame through ros
    # --------------------------------------------------------
    def base2head_callback(self, msg):
        T = np.eye(4)

        t = msg.pose.position
        T[:3, 3] = [t.x, t.y, t.z]

        q = msg.pose.orientation
        R = tft.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        T[:3, :3] = R

        self.base2head = T

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Publish base→QR TF using RealSense & ArUco")
    ap.add_argument('--intrinsics',  default="camIntrinsic.yaml", help="YAML with camera_matrix & dist_coeff")
    ap.add_argument('--width',       type=int, default=1280)
    ap.add_argument('--height',      type=int, default=720)
    ap.add_argument('--fps',         type=int, default=30)
    ap.add_argument('--marker-length',type=float, default=0.03)
    ap.add_argument('--dictionary',
                    choices=['4X4_50','5X5_100','6X6_50','7X7_1000'],
                    default='6X6_50')
    ap.add_argument('--base-frame',  default='base_link')
    ap.add_argument('--head-frame',   default='head_link')
    ap.add_argument('--qr-frame',    default='object_frame')
    ap.add_argument('--filter-N',    type=int, default=10,
                    help="window size for moving-average filter")
    ap.add_argument('--display',  action='store_true')
    args = ap.parse_args()

    QRPublisher(args).spin()


if __name__ == '__main__':
    main()
