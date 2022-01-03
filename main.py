import cv2
import numpy as np
import math


def detectCross(event, x, y, flags, params):
    global p_start, p_end, zdj, name, x_coords, y_coords, x_m, y_m
    if event == cv2.EVENT_LBUTTONDOWN:
        p_start = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        try:
            p_end = (x, y)
            roi_1 = zdj[p_start[1]:p_end[1], p_start[0]:p_end[0]]
            cv2.imwrite('krzyzyk2.png', roi_1)
            cv2.rectangle(zdj, p_start, p_end, (0, 0, 255), 10)
            cv2.imshow(name, zdj)
            x_m = p_end[0] - p_start[0]
            y_m = p_end[1] - p_start[1]
            x_coords.append(x_m)
            y_coords.append(y_m)
        except cv2.error:
            pass
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        try:
            x_mid, y_mid = checkPosition2(x, y)
            roi_2 = zdj[y_mid - 100:y_mid + 100, x_mid - 100:x_mid + 100]
            cv2.imwrite('krzyzyk2.png', roi_2)
            cv2.rectangle(zdj, (x_mid - 100, y_mid - 100), (x_mid + 100, y_mid + 100), (0, 255, 0), 10)
            cv2.imshow(name, zdj)
            x_coords.append(x_mid)
            y_coords.append(y_mid)
        except TypeError:
            pass
    elif event == cv2.EVENT_RBUTTONDOWN:
        try:
            roi_3 = zdj[y - 100:y + 100, x - 100:x + 100]
            cv2.imwrite('krzyzyk2.png', roi_3)
            cv2.rectangle(zdj, (x - 100, y - 100), (x + 100, y + 100), (0, 187, 255), 10)
            cv2.imshow(name, zdj)
            x_coords.append(x)
            y_coords.append(y)
        except TypeError:
            pass
    elif event == cv2.EVENT_MBUTTONDOWN:
        try:
            wzr = cv2.imread('krzyzyk2.png', 1)
            if checkPosition(x, y):
                coords = templateMatching(zdj, wzr, name)
                x_coords.append(coords[0])
                y_coords.append(coords[1])
                if len(x_coords) == 8:
                    affineTransformation(x_coords, y_coords)
            else:
                print("Za daleko od krzyżyka!")
        except FileNotFoundError:
            print("Brak pliku!")


def templateMatching(zdj, wzorzec, name):
    wzorzec = cv2.cvtColor(wzorzec, cv2.COLOR_BGR2GRAY)
    zdj_temp = cv2.cvtColor(zdj, cv2.COLOR_BGR2GRAY)
    zmienna = cv2.matchTemplate(zdj_temp, wzorzec, cv2.TM_SQDIFF_NORMED)
    min_value, max_value, min_window, max_window = cv2.minMaxLoc(zmienna)
    lg_naroznik = min_window
    pd_naroznik = (min_window[0] + wzorzec.shape[1], min_window[1] + wzorzec.shape[0])
    cv2.rectangle(zdj, lg_naroznik, pd_naroznik, 255, 10)
    cv2.imshow(name, zdj)
    return min_window[0] + wzorzec.shape[1] / 2, min_window[1] + wzorzec.shape[0] / 2


def checkPosition2(x, y):
    if x < 0.1 * zdj.shape[1] and y < 0.1 * zdj.shape[0]:
        return 291, 236
    elif 0.4 * zdj.shape[1] < x < 0.6 * zdj.shape[1] and y < 0.1 * zdj.shape[0]:
        return 4686, 194
    elif x > 0.9 * zdj.shape[1] and y < 0.1 * zdj.shape[0]:
        return 9086, 144
    elif x < 0.1 * zdj.shape[1] and 0.4 * zdj.shape[0] < y < 0.6 * zdj.shape[0]:
        return 335, 4637
    elif x > 0.9 * zdj.shape[1] and 0.4 * zdj.shape[0] < y < 0.6 * zdj.shape[0]:
        return 9126, 4545
    elif x < 0.1 * zdj.shape[1] and y > 0.9 * zdj.shape[0]:
        return 371, 9035
    elif 0.4 * zdj.shape[1] < x < 0.6 * zdj.shape[1] and y > 0.9 * zdj.shape[0]:
        return 4775, 8990
    elif x > 0.9 * zdj.shape[1] and y > 0.9 * zdj.shape[0]:
        return 9169, 8944


def checkPosition(x, y):
    if x < 0.1 * zdj.shape[1] and y < 0.1 * zdj.shape[0]:
        return True
    elif 0.4 * zdj.shape[1] < x < 0.6 * zdj.shape[1] and y < 0.1 * zdj.shape[0]:
        return True
    elif x > 0.9 * zdj.shape[1] and y < 0.1 * zdj.shape[0]:
        return True
    elif x < 0.1 * zdj.shape[1] and 0.4 * zdj.shape[0] < y < 0.6 * zdj.shape[0]:
        return True
    elif x > 0.9 * zdj.shape[1] and 0.4 * zdj.shape[0] < y < 0.6 * zdj.shape[0]:
        return True
    elif x < 0.1 * zdj.shape[1] and y > 0.9 * zdj.shape[0]:
        return True
    elif 0.4 * zdj.shape[1] < x < 0.6 * zdj.shape[1] and y > 0.9 * zdj.shape[0]:
        return True
    elif x > 0.9 * zdj.shape[1] and y > 0.9 * zdj.shape[0]:
        return True
    else:
        return False


def affineTransformation(xCoords, yCoords):
    try:
        X, Y = np.loadtxt('tlowe3.txt', usecols=(1, 2), unpack=True)
        A = []
        L = []
        for xi, yi, Xi, Yi in zip(xCoords, yCoords, X, Y):
            L.append([Xi])
            L.append([Yi])
            A.append([1, xi, yi, 0, 0, 0])
            A.append([0, 0, 0, 1, xi, yi])
        L = np.array(L)
        A = np.array(A)
        At = np.transpose(A)
        Xn = np.linalg.inv(At.dot(A)).dot(At.dot(L))
        Xn = np.transpose(Xn)
        m1 = np.array([[Xn[0, 0]], [Xn[0, 3]]])
        m2 = np.array([[Xn[0, 1], Xn[0, 2]], [Xn[0, 4], Xn[0, 5]]])
        x_tl = []
        y_tl = []
        x_v = []
        y_v = []
        for idx, (xi, yi, Xi, Yi) in enumerate(zip(xCoords, yCoords, X, Y)):
            coords_temp = np.array([[xi], [yi]])
            coords = m1 + m2.dot(coords_temp)
            X_tl = coords[0, 0]
            Y_tl = coords[1, 0]
            x_tl.append(X_tl)
            y_tl.append(Y_tl)
            x_v.append(Xi - X_tl)
            y_v.append(Yi - Y_tl)
            print('Wyliczone współrzędne tłowe punktu ' + str(idx + 1) + ': x = ' + str(X_tl) + ' y = ' + str(Y_tl))
        m0 = math.sqrt((sum(square(x_v)) + sum(square(y_v))) / 2)
        print('Wyznaczone parametry transformacji: a0 = ' + str(Xn[0, 0]) + ' a1 = ' + str(Xn[0, 1])
                + ' a2 = ' + str(Xn[0, 2]) + '\n' + ' b0 = ' + str(Xn[0, 3])
                + ' b1 = ' + str(Xn[0, 4]) + ' b2 = ' + str(Xn[0, 5]))
        for idx, (xv, yv) in enumerate(zip(x_v, y_v)):
            print('Błędy punktu ' + str(idx + 1) + ': x = ' + str(xv) + ' y = ' + str(yv))
        print('Wartość błędu wyrównania: ' + str(m0))
    except FileNotFoundError:
        print("Brak pliku!")


def square(arr):
    return map(lambda x: x ** 2, arr)

zdj = cv2.imread('724.tif', 1) # sciezka wzgledna tla
x_coords = []
y_coords = []
name = 'Nazwa okna 1'
cv2.namedWindow(name, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(name, detectCross)
cv2.imshow(name, zdj)
cv2.waitKey()
cv2.destroyAllWindows()