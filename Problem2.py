import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy

'''
This function is to outline circles belonging to each r value and the corresponding local max (a,b) values
'''
def add_max(B, img):
    for rab in B:
        center_coordinates = (rab[1], rab[2])
        color = (255, 0, 0) #BGR, want blue circles
        thickness = 1

        cv2.circle(img, center_coordinates, rab[0], color, thickness) #should work as canvas

    cv2.imshow('circled!', img)
    cv2.waitKey(0)

    return img
'''
The following function finds the local maxima/highest accumulated (a,b)-values

Our accumulator matrix is 3D, so we iterate over each element in the matrix and find the local maxima by querying
each element's neighborhood within the layer
'''
def findMax(A):
    max_list = []
    for r in range(A.shape[0]):
        for a in range(A.shape[1]):
            for b in range(A.shape[2]):
                max_val = np.max(A[r, a: a + 20, b: b + 20])
                max_list.append(max_val)
    max_list = list(set(max_list))

    '''
    we choose our final max values based on whether or not the value is above the 9xth percentile,
        indicating it is very likely true circle. 
    '''
    max_fin = []
    Q3 = np.percentile(max_list, 85, interpolation='midpoint')
    for value in max_list:
        if value > Q3:
            max_fin.append(value)

    '''
    Now we want to make a list of tuples containing the indices corresponding to the highest voted (r, a, b) values
    in the accumulator matrix
    '''
    index_list = []
    for element in max_fin:
        r, a, b = np.where(A == element)
        index_list.append((r[0], a[0], b[0]))
    print(index_list)

    '''
    Now we do a soft remove of any similar indices in case the bin sizes are not sufficient for filtering alike elements
    '''
    index_list = list(index_list)
    new_index_list = copy.deepcopy(index_list)
    print('before ', index_list)

    for i in range(len(new_index_list)):
        for j in range(i, len(new_index_list)):
            diff = np.subtract(new_index_list[i], new_index_list[j])
            diff[diff <= 25] = 0

            #we only need one of the similar values, so delete the other
            if 0 in diff:
                try:
                    index_list.remove(index_list[j])
                except:
                    continue

    print('after ', index_list)
    return index_list

'''
The  following function is to memoize the degree-to-radian calculations, as to not need to recompute it each time
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

#gradient is undefined at borders of image
def gradient_direction(x, y, img):
    if x == img.shape[1]: #in case at border
        x_grad = 0 #if at right edge point
    else:
        x_grad = img[x+1, y] - img[x,y]

    if y == 0:
        y_grad = 0# if at top of img
    else:
        y_grad = img[x, y+1] - img[x,y]

    grad = y_grad/x_grad
    g_direction = np.arctan(grad)

    return g_direction

def hough(edges, a_bin_list, b_bin_list):
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
    img_width = edges.shape[1] #get the width of the image
    img_height = edges.shape[0]

    #get diagonal length of image
    img_diag = np.int(np.floor(np.sqrt(np.square(img_height) + np.square(img_width))))
    cos, sin = preFill_radians()
    #accumulator
    A = np.zeros((50, img_width, img_height), dtype=int)
    print('size here ', A.shape)

    #make all data into 3d array for binning, matrix(r, a, b)
    #accumulator = np.zeros((20, img_height, img_width), dtype=int)
    for r in range(1, 50):
        print('r value', r)
        for x in range(edges.shape[0]):
            for y in range(edges.shape[1]):
                if edges[x][y] > 0: #if current point's value is > 0, it is an edge point
                    for theta in range(0, 361):
                        a = x-(r*cos[theta]) #just a table lookup and minor calculation
                        b = y-(r*sin[theta]) #just a table lookup and minor calculation

                        #conitinue to next loop without storing ab values, because if a or b is negative,
                        #the circle origin pixel locations are from off-screen/out of frame
                        if a < 0 or b < 0:
                            continue

                        #find the closest value from the bin list
                        a_bin_val = min(a_bin_list, key=lambda x: abs(x - a))
                        b_bin_val = min(b_bin_list, key=lambda x: abs(x - b))

                        A[r, a_bin_val, b_bin_val] += 1
                else:
                    continue #cont. to next iteration of current loop
    return A

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

    #generate a and b bins
    a_bin_list = []
    b_bin_list = []
    for i in range(0, img.shape[1], 50):
        a_bin_list.append(i)

    for i in range(0, img.shape[0], 50):
        b_bin_list.append(i)

    #perform Hough transform
    accum = hough(edges, a_bin_list, b_bin_list)
    print('acum', accum)

    #now find max (a,b)
    max_list = findMax(accum)
    print('max list', max_list)

    #lastly we want to add the circles corresponding to the found (a,b)
    fin_img = add_max(max_list, img)

    cv2.waitKey(0)

main()