# importacion de las librerias necesarias
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


# construccion y procesamiento de los argumentos que se insertaran desde la linea de comando
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path a la imagen de entrada")
ap.add_argument("-w", "--width", type=float, required=True,
                help="ancho de el objeto a la izquieda de la imagen (en pulgadas)")
args = vars(ap.parse_args())

# cargar la imagen, convertirla a escala de grises, y aplicar un poco de blur
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# realizar deteccion de bordes, despues aplicar dilatacion y erosion a la imagen
# para reducir las distancias entre los bordes de los objetos
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# encontrar los contornos
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# ordenar los contornos de izquierda a derecha e inicializar la variable de
# 'pixels per metric' que es la variable de calibracion
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

# recorrer los contornos
for c in cnts:
    # si es contorno no es lo suficientemente grande, se ignora
    if cv2.contourArea(c) < 100:
        continue

    # indetificar las cajas que delimitan cada objeto
    orig = image.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # ordenar los puntos de los contornos de las cajas, y
    # luego conectar cada punto para formar la caja
    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

    # recorrer cada punto para trazar las lineas
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

    # calcular el punto medio de los bordes, midpoint
    (tl, tr, br, bl) = box

    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # dibujar los puntos medios en la imagen
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # dibujar las lineas entre los puntos medios
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
             (255, 0, 255), 2)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
             (255, 0, 255), 2)

    # calcular la distancia euclidiana entre los puntos medios
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    # si la variable de calibracion no ha sido inicializada,
    # inicalizar, en pulagadas
    if pixelsPerMetric is None:
        pixelsPerMetric = dB / args["width"]

    # calcular las dimensiones de los objetos
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric

    # insertar el texto que indica las dimensiones de los objetos
    cv2.putText(orig, "{:.1f}in".format(dimA),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 0, 255), 2)
    cv2.putText(orig, "{:.1f}in".format(dimB),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 0, 255), 2)

    # mostrar la imagen final
    cv2.imshow("Image", orig)
    cv2.waitKey(0)
