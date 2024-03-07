import numpy
import os

######################################################################
### Parameter and arguments definition

DB_BFtemplatesA = "C:/Users/amith/Desktop/multibiometric/irisDB/"
DB_BFtemplatesB = "C:/Users/amith/Desktop/multibiometric/faceDB/"
fusionList = "C:/Users/amith/Desktop/multibiometric/input_files/fusionList.txt"
DB_BFtemplates_fused = 'C:/Users/amith/Desktop/multibiometric/BFtemplates_fused/'

if not os.path.exists(DB_BFtemplates_fused):
    os.mkdir(DB_BFtemplates_fused)

# nBits & nWords
N_BF_A = 32 # iris in the example
BF_SIZE_A = 1024
N_BF_B = 32*3*10 # face in the example
BF_SIZE_B = 16


####################################################################
# Fuses tempA and tempB according to the position vector pos
def fuse_BF_templates(BFtempA, BFtempB, pos):

    assert N_BF_A == BFtempA.shape[0], 'Dimensions from template A do not match: wrong number of Bloom filters'
    assert BF_SIZE_A == BFtempA.shape[1], 'Dimensions from template A do not match: wrong Bloom filter size'
    assert N_BF_B == BFtempB.shape[0], 'Dimensions from template B do not match: wrong number of Bloom filters'
    assert BF_SIZE_B == BFtempB.shape[1], 'Dimensions from template B do not match: wrong Bloom filter size'

    temp = BFtempA
    index = 0
    for i in range (N_BF_B):
        temp[pos[0][i], pos[1][i] : pos[1][i] + BF_SIZE_B] = numpy.bitwise_or(temp[pos[0][i], pos[1][i] : pos[1][i] + BF_SIZE_B], BFtempB[index, : ]) # ORing Iris & Face templates
        index += 1
    return temp

####################################################################
### Template fusion

# define position vector for the fusion
pos = numpy.zeros([2, N_BF_B], dtype=int) # Position vector
ratio = int(N_BF_B / N_BF_A) # we ensure equal number of BFs of B fused per BF of A

for i in range(N_BF_A):
    pos[0, i * ratio: (i+1)*ratio] = i * numpy.ones([1, ratio])            # BF where it will be allocated 
    pos[1, i * ratio: (i+1)*ratio] = range(0, BF_SIZE_B*ratio, BF_SIZE_B)  # position within the BF

index = 1
f = open(fusionList, 'r')
for filenames in f.readlines():
    print(filenames)

    r = filenames.split()
    tempA = numpy.loadtxt(DB_BFtemplatesA + r[0]).astype(int) #iris data
    tempB = numpy.loadtxt(DB_BFtemplatesB + r[1]).astype(int) #face data
    bfs = fuse_BF_templates(tempA, tempB, pos) #fused data
    numpy.savetxt(DB_BFtemplates_fused + str(index) + '_BFtemplate.txt', bfs, fmt='%d')

    index += 1
