from utilities import *
from Line import *

from moviepy.editor import VideoFileClip
from IPython.display import HTML

refresh=int('0x02', base=16)

#Points to warp and uwarp images
src = np.float32([[740,480],[1012,650],[308,650],[550,480]])
#src = np.float32([[668,440],[1012,650],[308,650],[610,440]])
dst = np.float32([[900,0],[900,650],[200,650],[200,0]])

rc,carPosition=0,0

#Calibrate camera
mtx, dist, rvecs, tvecs = Calibrate()

lines = Lines()

def ror(dat):
    if dat & 1==1: dat=dat<<8
    else: dat=dat>>1
    return dat

def processImage(image):
    #Undistort image
    #image=mpimg.imread("test_images/test6.jpg")
    undistort=cv2.undistort(image, mtx, dist, None, mtx)
    
    #Apply filters
    dummy, imgFiltered = ApplyFilters(undistort)
    
    #Warp
    imgWarped, dummy = warp(imgFiltered,src,dst)
    """
    plt.imshow(imgWarped,cmap='gray')
    plt.show()
    """
    #Find lanes
    
    #lines.testRun(imgWarped)
    lines.run(imgWarped)
    
    #colorImgWarped, dummy = warp(undistort)
    color_warp = lines.drawLines(imgWarped)
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp, dummy = warp(color_warp,dst,src)

    # Combine the result with the original image
    result = cv2.addWeighted(undistort, 1, newwarp, 0.3, 0)
    """
    #write radius and deviation from center in the image:
    #refresh=ror(refresh)
    rc=round(lines.getCurvature(imgWarped),1)
    carPosition=round(lines.getCarPosition(),3)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    cv2.putText(result,'Radius of curvature: '+str(rc),(10,100), font, 2,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(result,'Car position: '+str(carPosition),(10,150), font, 2,(255,255,255),2,cv2.LINE_AA)
    """
    #plt.imshow(result)
    #plt.show()
    
    return result

"""
for i in range(1,7):
    image=mpimg.imread("test_images/test"+str(i)+".jpg")
    result=processImage(image)
    plt.imshow(result)
    plt.show()

"""
projectlines_output = 'projectlines_output.mp4'
#clip1 = VideoFileClip("project_video.mp4").subclip(0,5)
clip1 = VideoFileClip("project_video.mp4")
#clip1 = VideoFileClip("challenge_video.mp4")
clip = clip1.fl_image(processImage) 
clip.write_videofile(projectlines_output, audio=False)
#"""
