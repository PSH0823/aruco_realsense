import rospy
import numpy as np
from geometry_msgs.msg import Twist
from pynput import keyboard

class ArucoKeyboardTeleop:
    def __init__(self):
        self.pressed = set()

        self.lin_vel = np.zeros(3)   # Linear velocity (vx, vy, vz)
        self.ang_vel = np.zeros(3)   # Angular velocity (roll, pitch, yaw)

        self.VEL = 0.5               # m/s
        self.ANG_VEL = 0.1           # rad/s

    def on_press(self, key):
        self.pressed.add(key)
        self.update_velocity()

    def on_release(self, key):
        self.pressed.discard(key)
        self.update_velocity()

    def update_velocity(self):
        self.lin_vel[:] = 0.0 

        if keyboard.Key.left in self.pressed:
            self.lin_vel[1] = self.VEL
        if keyboard.Key.right in self.pressed:
            self.lin_vel[1] = -self.VEL
        if keyboard.Key.up in self.pressed:
            self.lin_vel[2] = self.VEL
        if keyboard.Key.down in self.pressed:
            self.lin_vel[2] = -self.VEL

        for key in self.pressed:
            try:
                if key.char == 'w':
                    self.lin_vel[0] = -self.VEL
                elif key.char == 's':
                    self.lin_vel[0] = self.VEL
            except AttributeError:
                pass

    def run(self):
        rospy.init_node("aruco_velocity_commander")
        pub = rospy.Publisher("/aruco_velocity", Twist, queue_size=1)

        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            msg = Twist()
            msg.linear.x = self.lin_vel[0]
            msg.linear.y = self.lin_vel[1]
            msg.linear.z = self.lin_vel[2]
            pub.publish(msg)
            rate.sleep()

if __name__ == "__main__":
    teleop = ArucoKeyboardTeleop()
    teleop.run()
