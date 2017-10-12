import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

def Calibrate():
    objpoints, imgpoints =[], []
    objp=np.zeros((6*9,3),np.float32)
    objp[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)
    
    images=glob.glob('camera_cal/calibration*.jpg')
    gray=0
    for image in images:
        img = mpimg.imread(image)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        
        if ret==True:
            imgpoints.append(corners)
            objpoints.append(objp)
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist, rvecs, tvecs


def warp(img,src,dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]))
    return warped, M

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for line in lines:
        x1,y1,x2,y2 = line
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    result = weighted_img(line_img, img, α=0.7, β=1., λ=0.)
    return result



def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):    
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobel = cv2.Sobel(gray, cv2.CV_64F, orient=='x', orient=='y', ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    binary_output = np.copy(sbinary) 
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    abs_sobel=np.absolute(np.sqrt(sobelx**2 + sobely**2))
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a binary mask where mag thresholds are met
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    binary_output = np.copy(sbinary) 
    return binary_output
    
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    graddir = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    gbinary = np.zeros_like(graddir)
    gbinary[(graddir >= thresh[0]) & (graddir <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    binary_output = np.copy(gbinary) 
    return binary_output
    
def S_threshold(img,thresh=(0,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    S = hls[:,:,2]
    sbinary = np.zeros_like(S)
    cS1=255/float(np.amax(S)-np.amin(S))
    S1=np.array(list(map(lambda e: (e-np.amin(S))*cS1, S)),np.float32)
    sbinary[(S1 > thresh[0]) & (S1 <= thresh[1])] = 1
    #sbinary[(S > thresh[0]) & (S <= thresh[1])] = 1
    return sbinary
    

def ApplyFilters(image, ksize=3):
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3)) #(0.7,1.3)

    combinedSobel = np.zeros_like(dir_binary)
    combinedSobel[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    s_binary = S_threshold(image, thresh=(120,255))

    color_binary = np.dstack(( np.zeros_like(combinedSobel), combinedSobel, s_binary))
    
    combined = np.zeros_like(combinedSobel)
    combined[(s_binary == 1) | (combinedSobel == 1)] = 1
    
    return color_binary, combined
        



def testCalibration():
    mtx, dist, rvecs, tvecs = Calibrate()

    chess=mpimg.imread("camera_cal/calibration1.jpg")
    undist_chess=cv2.undistort(chess, mtx, dist, None, mtx)
    
    #f, axarr = plt.subplots(2, sharex=True)
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(chess)
    axarr[0].set_title('Original image')
    axarr[1].imshow(undist_chess)
    axarr[1].set_title("Undistorted image")
    f.show()
    f.savefig("chess.jpg", bbox_inches='tight')
    
    test=mpimg.imread("test_images/test1.jpg")
    undist_test=cv2.undistort(test, mtx, dist, None, mtx)
    
    e, axarr1 = plt.subplots(1,2)
    axarr1[0].imshow(test)
    axarr1[0].set_title('Original image')
    axarr1[1].imshow(undist_test)
    axarr1[1].set_title("Undistorted image")
    e.show()
    e.savefig("test.jpg", bbox_inches='tight')


def testIndFilters( ksize=3):
    mtx, dist, rvecs, tvecs = Calibrate()
    
    #image=mpimg.imread("test_images/test1.jpg")
    image=mpimg.imread("signs_vehicles_xygrad.png")
    img=cv2.undistort(image, mtx, dist, None, mtx)
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

    f, axarr = plt.subplots(2,2)
    ax= axarr.flatten()
    ax[0].imshow(gradx)
    ax[0].set_title('Sobel x')
    ax[1].imshow(grady)
    ax[1].set_title("sobel y")
    ax[2].imshow(mag_binary)
    ax[2].set_title("Magnitude")
    ax[3].imshow(dir_binary)
    ax[3].set_title("Direction")
    f.show()
    f.savefig("testIndFilters.jpg", bbox_inches='tight')
    
def testFilters():
    mtx, dist, rvecs, tvecs = Calibrate()
    
    image=mpimg.imread("signs_vehicles_xygrad.png")
    #image=mpimg.imread("test_images/test1.jpg")
    img=cv2.undistort(image, mtx, dist, None, mtx)
    color_binary, combined = ApplyFilters(img)

    f, (ax1, ax2) = plt.subplots(1,2)
    ax1.set_title('Stacked thresholds')
    ax1.imshow(color_binary)
    
    ax2.set_title('Combined S channel and gradient thresholds')
    ax2.imshow(combined, cmap='gray')
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    f.show()
    f.savefig("testFilters.jpg", bbox_inches='tight')

def testWarp():
    image=mpimg.imread("test_images/straight_lines2.jpg")
    mtx, dist, rvecs, tvecs = Calibrate()
    img=cv2.undistort(image, mtx, dist, None, mtx)

    imgInit = np.copy(img) 
    srcLines=[[740,480,1012,650],[308,650,550,480]]
    imgInit = draw_lines(imgInit,srcLines)
    """    
    dpi = 100
    height, width = img.shape[0:2]
    figsize = width/float(dpi), height/float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(imgInit, interpolation='nearest')
    ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)
    fig.savefig('imgInit.jpg', dpi=dpi, transparent=True)
    plt.show()
    """
    
    #plt.figure()
    warped, M = warp(img)
    dstLines=[[900,0,900,650],[200,650,200,0]]
    imgFinal = draw_lines(warped,dstLines)
    """
    plt.imshow(imgFinal)
    plt.show()
    """
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
    ax1.set_title('Initial image')
    ax1.imshow(imgInit)
    
    ax2.set_title('Warped image')
    ax2.imshow(imgFinal)
    #plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    f.show()
    f.savefig("warp.jpg", bbox_inches='tight')
