import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy

'''
This function is to outline circles belonging to each r value and the corresponding local max (a,b) values
'''
def add_max(B, img):
    for key in B.keys():
        #get r value
        r = int(float(key.split(',')[0]))
        #get a value
        a = int(float(key.split(',')[1]))
        #get b value
        b = int(float(key.split(',')[2]))
        print('rab',r, a,b)

        center_coordinates = (a, b)
        color = (255, 0, 0) #BGR, want blue circles
        thickness = 1

        cv2.circle(img, center_coordinates, r, color, thickness) #should work as canvas

    cv2.imshow('circled!', img)
    cv2.waitKey(0)
    '''
    THIS STILL IS WRONG
    '''

    return img
'''
The following function finds the local maxima/highest accumulated (a,b)-values, top 5 overall
'''
def findMax(A, width):
    #A = {r1: {(a1,b1):votes, (a2,b2):votes, ...}, r2: {}, ...}
    r_max_dict = {}
    for r in range(1, np.int(width/4)):
        '''
            1-choose an r value, access the corresponding dictionary
            2-find top 10 highest/local max counts per (a,b) combination
            3-find top 10 local max counts overall
        '''
        #find top votes per r-value, then delete key value pair and find next top value:
        #   repeat for 10
        for count in range(10): #top 10 vote-(a,b) pairs
            max_ab = max(A[r].values())
            for key in A[r].keys(): #keys are (a,b) value combinations
                if A[r][key] == max_ab:
                    max_key = str(r) + ',' + key
                    r_max_dict[max_key] = A[r][key]
                    del A[r][key] #delete key value pair
                    break

    #r_max_dict = {'r,a,b' = votes, ...}
    while (len(r_max_dict) > 5):
        del r_max_dict[min(r_max_dict, key=r_max_dict.get)]

    return r_max_dict

'''
The following function is to memoize the degree-to-radian calculations, as to not need to recompute it each time
    on the fly. This will reduce the calculation time to O(1) constant-time lookup
    
3 hash tables/dicts are used:
    1) one table to hold all radian equivalents of 0-360deg to rad
    2) another table to hold all cos(theta) values for all radians
    3) a last table to hold all sin(theta) values for all radians
'''
def preFill_radians():
    rads = {}
    cos_calcs = {}
    sin_calcs = {}

    #to calculate all radian-degree equivalencies
    for theta in range(0, 361):
        rads[theta] = np.deg2rad(theta)

    #sin and cos
    for theta in rads.keys():
        cos_calcs[theta] = np.cos(rads[theta])
        sin_calcs[theta] = np.sin(rads[theta])
    return cos_calcs, sin_calcs

def hough(edges):
    '''
    To generate all possible circle origins corresponding to one (x,y) edge point in the original image:
        1) take a detected edge point from image space
        2) extract its x and y values
        3) generate all possible (a,b) circle origin-point combinations that could have produced that (x,y),
            and store each combination in the accumulator matrix (dictionary), adding one for each instance of an
            (a,b) pair:
                a = x-cos(theta), b = y-sin(theta), iterate over all theta
        Repeat steps 1-3 until we have iterated over all pixels
    '''
    A = {}  # accumulator 'matrix' (dictionary)
    img_width = edges.shape[1] #get the width of the image
    img_height = edges.shape[0]
    cos, sin = preFill_radians()
    for r in range(1, np.int(img_width / 4)):
        print('r value', r)
        for x in range(edges.shape[0]):
            for y in range(edges.shape[1]):
                if edges[x][y] > 0: #if current point's value is > 0, it is an edge point
                    for theta in range(0, 361):
                        a = np.rint(x-(r*cos[theta])) #just a table lookup and minor calculation
                        b = np.rint(y-(r*sin[theta])) #just a table lookup and minor calculation

                        #conitinue to next loop without storing ab values, because if a or b is negative,
                        #the circle origin pixel locations are from off-screen/out of frame
                        if a < 0 or b < 0:
                            continue

                        ab_str = str(a)+','+str(b)
                        #print(ab_str)

                        if ab_str in A.keys():
                            A[ab_str] += 1
                        else:
                            A[ab_str] = 1
                else:
                    continue #cont. to next iteration of current loop
def main():
    #load the image
    img = cv2.imread('/Users/richardmagnotti/PycharmProjects/MVHW3/csc249tracking/ueCwZ.png')
    img2 = copy.deepcopy(img)
    cv2.imshow('img1', img)
    cv2.waitKey(0)
    
    #running median filter over image to get rid of noisy/unnecessary edges
    img2 = cv2.medianBlur(img2, 5)

    #detect edges w/ Canny
    edges = cv2.Canny(img2, 120, 180)
    print(edges.shape)
    cv2.imshow('imgCanny', edges)
    cv2.waitKey(0)

    #perform Hough transform
    accum = hough(edges)
    print('acum', accum)

    #now find max (a,b) accumulated votes per r value
    max_list = findMax(accum, img.shape[1])
    print('max list', max_list)

    #lastly we want to add the circles corresponding to the found (a,b) and r values back to the image
    fin_img = add_max(max_list, img)

main()