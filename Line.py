import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from functools import reduce

class Line:
    def __init__(self, nwindows=9, margin=100, minpix=50, hand="left", nIterations=15):
        self.current_fit=[]
        self.nIterations=nIterations
        self.detected = False                           
        #self.recent_xfitted = []                       # x values of the last n fits of the line 
        #self.bestx = None                               #average x values of the fitted line over the last n iterations
        self.best_fit = np.array([0,0,0], dtype='float')                            #polynomial coefficients averaged over the last n iterations
        self.old_fits = None                              #polynomial coefficients of old fits
        self.nwindows = nwindows
        self.margin=margin
        self.minpix=minpix
        self.hand=hand
        self.radius_of_curvature = None
        #self.line_base_pos = None                        #distance in meters of vehicle center from the line
        #self.diffs = np.array([0,0,0], dtype='float')    #difference in fit coefficients between last and new fits
        self.allx = None                                 #x values for detected line pixels
        self.ally = None                                 #y values for detected line pixels
    
    def getBase(self,img):
        #histogram of the bottom half of the image
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        
        #find the peaks of the two halves of the histogram
        midpoint = np.int(histogram.shape[0]/2)
        
        offset=0
        if self.hand=="right": offset=midpoint
        x_base = np.argmax(histogram[midpoint:]) + offset
        
        return x_base

    def slidingWindows(self,img):
        x_base = self.getBase(img)
        window_height = np.int(img.shape[0]/self.nwindows)
        
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        #out_img = np.dstack((img, img, img))*255

        # Current positions to be updated for each window
        x_current = x_base

        # Create empty lists to receive left and right lane pixel indices
        lane_inds = []
        
        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            win_x_low = x_current - self.margin
            win_x_high = x_current + self.margin
            
            # Draw the windows on the visualization image
            #cv2.rectangle(out_img,(win_x_low,win_y_low),(win_x_high,win_y_high),(0,1,0), 2)
            
            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]
            
            # Append these indices to the lists
            lane_inds.append(good_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > self.minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))
                
        # Concatenate the arrays of indices
        lane_inds = np.concatenate(lane_inds)

        return lane_inds
        
    def getLine(self, img):
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        if not self.detected:
            lane_inds = self.slidingWindows(img)
        else:
            lane_inds = ((nonzerox > (self.current_fit[0]*(nonzeroy**2) + self.current_fit[1]*nonzeroy + self.current_fit[2] - self.margin)) & (nonzerox < (self.current_fit[0]*(nonzeroy**2) + self.current_fit[1]*nonzeroy + self.current_fit[2] + self.margin)))
            
        self.allx = nonzerox[lane_inds]
        self.ally = nonzeroy[lane_inds] 

        # Fit a second order polynomial to each
        self.current_fit = np.polyfit(self.ally, self.allx, 2)
        
        return self.current_fit

    
    def plotLineWindow(self, img, window_img=None):
        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
        fitx = self.current_fit[0]*ploty**2 + self.current_fit[1]*ploty + self.current_fit[2]
        
        # Create an image to draw on and an image to show the selection window
        if len(img.shape)<3: out_img = np.dstack((img, img, img))
        else: out_img = img
        if window_img==None: window_img = np.zeros_like(out_img)

        # Color in left and right line pixels
        if self.hand=="left": out_img[self.ally, self.allx] = [1, 0, 0]
        else: out_img[self.ally, self.allx] = [0, 0, 1]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        line_window1 = np.array([np.transpose(np.vstack([fitx-self.margin, ploty]))])
        line_window2 = np.array([np.flipud(np.transpose(np.vstack([fitx+self.margin, ploty])))])
        line_pts = np.hstack((line_window1, line_window2))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([line_pts]), (0,255, 0))
        #result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        return out_img, window_img, ploty, fitx

    
    def getRadius(self, img, current=True):
        """
          * Current=True - calculates the radius of the current measurement and it is used only to sanity checks
          * Current=False - calculates the radius from the average parameters of the last measurement
        """
        # Define y-value where we want radius of curvature
        # Maximum y-value, corresponding to the bottom of the image
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
        y_eval = np.max(ploty)
        if current:
            fitx = self.current_fit[0]*ploty**2 + self.current_fit[1]*ploty + self.current_fit[2]
        else:
            fitx = self.best_fit[0]*ploty**2 + self.best_fit[1]*ploty + self.best_fit[2]
            
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(ploty*ym_per_pix, fitx*xm_per_pix, 2)

        # Calculate the new radii of curvature
        rc = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        
        if not current:
            self.radius_of_curvature=rc
            # Now our radius of curvature is in meters
            #print(self.radius_of_curvature, 'm')
        return rc

    
    def reset(self):
        self.detected=False
        self.best_fit = np.array([0,0,0], dtype='float')                            
        self.old_fits = None
        return

    def saveFit(self):
        if self.old_fits==None:
            self.old_fits=np.array([self.current_fit], dtype='float')
            return
        if len(self.old_fits)>=self.nIterations:
            self.old_fits=np.delete(self.old_fits,(0),axis=0)
        self.old_fits = np.append(self.old_fits,[self.current_fit],axis=0)

        self.smooth()
        return
    
    def smooth(self):
        self.best_fit = self.old_fits.mean(axis=0)
        return self.best_fit

    def getBestFit(self):
        return self.best_fit
    

class Lines:
    def __init__(self, nwindows=9, margin=100, minpix=50, nIterations=25):
        self.theLines=[]
        for h in ["left","right"]:
            self.theLines.append(Line(hand=h, nwindows=nwindows, margin=margin, minpix=minpix, nIterations=nIterations))
        self.oldSanityChecks=[]
        self.nIterations=nIterations

    def getLines(self,img):
        for l in self.theLines:
            l.getLine(img)
        return
    
    def plotLinesWindow(self,img):
        lout_img, window_img, ploty, lfitx = self.theLines[0].plotLineWindow(img)
        out_img, window_img, ploty, rfitx = self.theLines[1].plotLineWindow(lout_img, window_img)
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
        plt.imshow(out_img)
        plt.plot(lfitx, ploty, color='yellow')
        plt.plot(rfitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()
        return
    
    def checkCurvature(self, img):
        radius=[]
        for l in self.theLines:
            radius+=l.getRadius(img, current=True)

        if abs(radius[0]-radius[1])>(radius[0]+radius[1])/4.:
            return False
        return True

    def getCurvature(self,img):
        radius=[]
        for l in self.theLines:
            radius+=[l.getRadius(img, current=False)]
        rad=np.mean(radius)
        return rad

    def saveFit(self, img):
        for l in self.theLines:
            l.saveFit()
        return 

    def getLineSeparation(self):
        xm_per_pix = 3.7/700
        lineSeparation=abs(self.theLines[0].current_fit[2]-self.theLines[1].current_fit[2])*xm_per_pix
        return lineSeparation

    def getCarPosition(self):
        xm_per_pix = 3.7/700
        centre=1280/2.
        deviation = ((centre - self.theLines[0].current_fit[2]) - (self.theLines[1].current_fit[2]-centre))/2. * xm_per_pix
        return deviation

    def checkParallelism(self,img):
        xm_per_pix = 3.7/700
        ploty = np.linspace(0, img.shape[0]-1, 10 )
        points=[]
        for l in self.theLines:
            points.append( l.current_fit[0]*ploty**2 + l.current_fit[1]*ploty + l.current_fit[2])
        distances=list(map(lambda x,y:(x-y)*xm_per_pix, points[0], points[1]))
        m=np.mean(distances)
        std=np.std(distances)
        if False in map(lambda d: abs(d-m)<3*std, distances): return False
        return True
            
    def sanityCheck(self,img):
        if abs(self.getLineSeparation()-3.7)>0.5: return False
        if abs(self.getCarPosition())>1: return False
        if not self.checkParallelism(img): return False
        return True

    def watchDog(self,check):
        if len(self.oldSanityChecks)>=self.nIterations:
            self.oldSanityChecks=self.oldSanityChecks[1:]
        self.oldSanityChecks.append(check)
        if self.oldSanityChecks.count(False)>self.nIterations*3/2:
            self.reset()
        return

    def reset(self):
        self.oldSanityChecks=[]
        for l in self.theLines:
            l.reset()
        return
            
    def run(self,img):
        self.getLines(img)
        check=self.sanityCheck(img)
        if check: self.saveFit(img)
        self.watchDog(check)
        return

    def testRun(self,img):
        self.getLines(img)
        check=self.sanityCheck(img)
        if check: self.saveFit(img)
        self.watchDog(check)
        self.plotLinesWindow(img)
        self.reset()
        return

    def drawLines(self,imgWarped):
        warp_zero = np.zeros_like(imgWarped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        ploty = np.linspace(0, imgWarped.shape[0]-1, imgWarped.shape[0] )
        lfitx = self.theLines[0].best_fit[0]*ploty**2 + self.theLines[0].best_fit[1]*ploty + self.theLines[0].best_fit[2]
        rfitx = self.theLines[1].best_fit[0]*ploty**2 + self.theLines[1].best_fit[1]*ploty + self.theLines[1].best_fit[2]
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([lfitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([rfitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        return color_warp
