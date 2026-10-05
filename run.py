import head_pose
import detect
import threading as th


if __name__ == "__main__":
    #main()
    head_pose_thread = th.Thread(target=head_pose.pose)
    detect_thread = th.Thread(target=detect.run_detection)

    head_pose_thread.start()
    detect_thread.start()

    head_pose_thread.join()
    detect_thread.join()

