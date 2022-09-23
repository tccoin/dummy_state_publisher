import numpy as np
import cv2 
import trajectory_transform as tt
folderPath = "/home/link/Projects/curly_slam/data/soulcity"
imageFolder = "/depth_left/"
imageIndex = 0
savePath = "/home/link/Projects/curly_slam/data/soulcity/depth_image/"
trajName = "pose_left.txt"
def readimage():
    global imageIndex
    imageName = '0'*(6 - len(str(imageIndex))) + str(imageIndex) + "_left_depth.npy"
    imageName = folderPath + imageFolder + imageName
    print(imageName)
    depth = np.load(imageName)
    #depth = np.uint16(depth * 1000)
    print(depth)
    
    cv2.imwrite(savePath + '0'*(6 - len(str(imageIndex))) + str(imageIndex) + "_left_depth.tiff",depth)
    imageIndex += 1
    depth8U = np.int8(depth/1000)
    depth8U = 1.0/50 * depth8U
    # cv2.imshow("test", depth8U)
    # cv2.waitKey(0)
def readTraj():
    traj = np.loadtxt(folderPath + "/" + trajName)
    traj = tt.ned2cam(traj)
    np.savetxt(folderPath + "/" + "newPose.txt",traj)
def main(): 
    readTraj()
    while(1):
        readimage()
    
if __name__=="__main__":
    main()
