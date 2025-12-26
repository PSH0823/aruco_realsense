import tf.transformations as tft
import yaml
import numpy as np

def load_yaml_head2cam_rot_xyz_fixed_frame(path):
    with open(path, 'r') as f:
        d = yaml.safe_load(f)
    euler_zyx = d["euler_zyx"]
    R = tft.euler_matrix(euler_zyx[0], euler_zyx[1], euler_zyx[2])[:3,:3]

    # quat = tft.quaternion_from_euler(euler_zyx[0], euler_zyx[1], euler_zyx[2])
    # print("Quaternion from fixed frame (xyz):", quat)

    # data = {
    #     'quaternion': {
    #         'x': float(quat[0]),
    #         'y': float(quat[1]),
    #         'z': float(quat[2]),
    #         'w': float(quat[3])
    #     },
    #     'rotation_matrix': R.tolist()
    # }

    # with open('orientation.yaml', 'w') as f:
    #     yaml.safe_dump(data, f, sort_keys=False)

    return R

def load_yaml_head2cam_rot_zyx_euler(path):
    with open(path, 'r') as f:
        d = yaml.safe_load(f)
    euler_zyx = d["euler_zyx"]
    R = tft.euler_matrix(euler_zyx[2], euler_zyx[1], euler_zyx[0], 'rzyx')[:3,:3]

    return R

def load_yaml_head2cam_rot_quat(path):
    with open(path, 'r') as f:
        d = yaml.safe_load(f)
    quat = d["quaternion"]  # [x, y, z, w]
    print("Quaternion from yaml:", quat)
    R = tft.quaternion_matrix([quat[0], quat[1], quat[2], quat[3]])[:3,:3]

    return R

def load_yaml_rot(path):
    with open(path, 'r') as f:
        d = yaml.safe_load(f)
    R = np.array(d["rotation_matrix"])

    return R

def main():
    path = 'TF_head2cam.yaml'
    R_fixed = load_yaml_head2cam_rot_xyz_fixed_frame(path)
    R_euler = load_yaml_head2cam_rot_zyx_euler(path)
    R_quat = load_yaml_head2cam_rot_quat(path)
    R_rot = load_yaml_rot(path)
    print("R from fixed frame (xyz):\n", R_fixed)
    print("R from euler (zyx):\n", R_euler)
    print("R from quaternion:\n", R_quat)
    print("R from saved rotation matrix:\n", R_rot)

    R_cv2mjc = np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, -1]])
    R_mujo = R_rot@R_cv2mjc
    print("R for mujoco:\n", R_mujo)
    R4 = np.eye(4)
    R4[:3, :3] = R_mujo
    q = tft.quaternion_from_matrix(R4)
    print(q)

    data = {
        'quaternion': {
            'x': float(q[0]),
            'y': float(q[1]),
            'z': float(q[2]),
            'w': float(q[3])
        }
    }
    
    with open('orientation_mujoco.yaml', 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)

if __name__ == "__main__":
    main()