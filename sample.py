import cv2
import numpy as np

def circle_mask(shape,x,y,r):
    canvas = np.zeros(shape,dtype=np.uint8)
    cv2.circle(canvas,(x,y),r,1,thickness=-1)
    return canvas

def hsv_mask(img_hsv):
    '''
    return red_mask,yellow_mask,green_mask
    '''
    sv_mask = img_hsv[:,:,1:]>100
    sv_mask = sv_mask[:,:,0]*sv_mask[:,:,1]
    m1 = img_hsv[:,:,0]<11
    m2 = img_hsv[:,:,0]>160
    red_mask = (m1+m2)*sv_mask
    yellow_mask = ((img_hsv[:,:,0]>=11)*(img_hsv[:,:,0]<50))*sv_mask
    green_mask = ((img_hsv[:,:,0]>=50)*(img_hsv[:,:,0]<90))*sv_mask
    return red_mask,yellow_mask,green_mask

def color_classifier(img,mask,hsv=True):
    '''
    red->0
    yellow->1
    green->2
    '''
    img_masked = img*mask
    if(hsv):
        red_mask,yellow_mask,green_mask = hsv_mask(img_masked)
        color_mask = {
            'red':red_mask,
            'yellow':yellow_mask,
            'green':green_mask,
        }
        color_score = np.array([])
        for i in color_mask.values():
            color_score = np.append(color_score,np.sum(i))
        print(color_score)
        return np.argmax(color_score)

    else:
        color_vector = {
            'red':np.array([0.,0.,1.]),
            'yellow':np.array([0.,0.707,0.707]),
            'green':np.array([0.,1.,0.]),
        }
        color_score = np.array([])
        for i in color_vector.values():
            color_score = np.append(color_score,np.sum(i))
        print(color_score)
        return np.argmax(color_score)

def getTrafficLight(img,dp=1.5,minDist=20,param1=100,param2=0,minRadius=0,maxRadius=30):
    '''
    Parameters
    ----------
    @param dp: Inverse ratio of the accumulator resolution to the image resolution. For example, if dp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has half as big width and height. For #HOUGH_GRADIENT_ALT the recommended value is dp=1.5, unless some small very circles need to be detected.

    @param minDist: Minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.

    @param param1: First method-specific parameter. In case of #HOUGH_GRADIENT and #HOUGH_GRADIENT_ALT, it is the higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller). Note that #HOUGH_GRADIENT_ALT uses #Scharr algorithm to compute image derivatives, so the threshold value shough normally be higher, such as 300 or normally exposed and contrasty images.

    @param param2: Second method-specific parameter. In case of #HOUGH_GRADIENT, it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first. In the case of #HOUGH_GRADIENT_ALT algorithm, this is the circle "perfectness" measure. The closer it to 1, the better shaped circles algorithm selects. In most cases 0.9 should be fine. If you want get better detection of small circles, you may decrease it to 0.85, 0.8 or even less. But then also try to limit the search range [minRadius, maxRadius] to avoid many false circles.

    @param minRadius: Minimum circle radius.

    @param maxRadius: Maximum circle radius. If <= 0, uses the maximum image dimension. If < 0, #HOUGH_GRADIENT returns centers without finding the radius. #HOUGH_GRADIENT_ALT always computes circle radiuses.
    
    Return
    ----------
    img,result
    '''

    #Create the color filter by HSV
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    red_mask,yellow_mask,green_mask = hsv_mask(img_hsv)
    color_mask = red_mask+yellow_mask+green_mask

    #find the circle
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray,(5,5),1)
    circles = cv2.HoughCircles(img_blur,method=cv2.HOUGH_GRADIENT_ALT,dp=dp,minDist=minDist,param1=param1,param2=param2,minRadius=minRadius,maxRadius=maxRadius)
    circles = np.array(circles[:,0])

    #Compare L1 distance to eliminate concentric circles
    L1_minDist = 15
    try:
        idx = -1
        for i in circles:
            x = int(i[0])
            y = int(i[1])
            r = int(i[2])
            idx = idx+1
            mask = color_mask*circle_mask(img.shape[0:2],x,y,r)
            if(np.sum(mask)/(np.pi*r**2)<0.3):
                circles = np.delete(circles,idx,0)
                idx = idx-1
            else:
                tmp = circles-circles[idx]
                tmp[:,:2] = np.abs(tmp[:,:2])
                overlap_idx = np.argwhere((tmp[:idx,0]+tmp[:idx,1])<L1_minDist)
                if(overlap_idx.shape[0]>0):
                    if(tmp[overlap_idx[0],2]<0):
                        circles = np.delete(circles,overlap_idx[0],0)
                        idx = idx-1
                        #cv2.circle(img,(x,y),r,(255,0,0),thickness=2)
                    else:
                        circles = np.delete(circles,idx,0)
                        idx = idx-1
                else:
                    #cv2.circle(img,(x,y),r,(255,0,0),thickness=2)
                    pass
        #result = [[(x,y),color],...]
        result = []
        for i in circles:
            x = int(i[0])
            y = int(i[1])
            r = int(i[2])
            mask = (circle_mask(img.shape[0:2],x,y,r)*color_mask)[:,:,None]
            color = color_classifier(img_hsv,mask)
            color_dict = {
                0:'red',
                1:'yellow',
                2:'green'
            }
            color = color_dict[color]
            result.append([(x,y),color])
            cv2.circle(img,(x,y),r,(255,0,0),thickness=2)
            cv2.putText(img,color,(x,y),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),thickness=1)

        return img, result
        
    except Exception:
        print( Exception.__traceback__())

if __name__ == '__main__':
    #2 6 7 fail
    img = cv2.imread(r'./trafficlight/test9.jpg')
    img_result,result = getTrafficLight(img.copy())
    #result = [[(x,y),color],...]
    print(result)
    cv2.imshow("Origin",img)
    cv2.imshow("Result",img_result)
    cv2.waitKey(0)
