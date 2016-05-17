from MultiNEAT.viz import DrawPhenotype
import cv2
import numpy
import MultiNEAT as NEAT


def DrawNet(net):
    # draw the phenotype
    img = numpy.zeros((250, 250, 3), dtype=numpy.uint8)
    img += 10
    DrawPhenotype(img, (0, 0, 250, 250), net )
    cv2.imshow("current best", img)
    cv2.waitKey(1)    
    
def DrawGene(gene):
    net = NEAT.NeuralNetwork()
    gene.BuildPhenotype(net)
    
    # draw the phenotype
    img = numpy.zeros((250, 250, 3), dtype=numpy.uint8)
    img += 10
    DrawPhenotype(img, (0, 0, 250, 250), net )
    cv2.imshow("current best", img)
    cv2.waitKey(1)
    