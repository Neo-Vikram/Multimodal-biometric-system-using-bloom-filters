import numpy
import math
import os
import argparse

######################################################################
### Parameter and arguments definition

# location of source templates and score files
#parser = argparse.ArgumentParser(description='Compute protected Bloom filter scores from a given DB and protocol.')

#parser.add_argument('DB_BFtemplates', help='directory where the protected BF templates are stored', type=str)
#parser.add_argument('matedComparisonsFile', help='file comprising the mated comparisons to be carried out', type=str)
#parser.add_argument('nonMatedComparisonsFile', help='file comprising the non-mated comparisons to be carried out', type=str)
#parser.add_argument('--scoresDir', help='directory where unprotected and protected scores will be stored', type=str, nargs='?', default = './scores/')
#parser.add_argument('--matedScoresFile', help='file comprising the mated scores computed', type=str, nargs='?', default = 'matedScoresBF.txt')
#parser.add_argument('--nonMatedScoresFile', help='file comprising the non-mated scores computed', type=str, nargs='?', default = 'nonMatedScoresBF.txt')

#args = parser.parse_args()
#DB_BFtemplates = args.DB_BFtemplates
DB_BFtemplates = 'C:/Users/amith/Desktop/multibiometric/BFtemplates_fused/'
#matedComparisonsFile = args.matedComparisonsFile
matedComparisonsFile = 'matedcomparison_file'
#nonMatedComparisonsFile = args.nonMatedComparisonsFile
nonMatedComparisonsFile = 'nonmatedcomparison_file'
#scoresDir = args.scores
scoresDir = 'Scores'
#matedScoresFile = args.matedScoresFile
matedScoresFile = 'matedscores_file'
#nonMatedScoresFile = args.nonMatedScoresFile
nonMatedScoresFile = 'nonMatedScores_file'

if not os.path.exists(scoresDir):
    os.mkdir(scoresDir)

####################################################################
### Some auxiliary functions

def hamming_distance(X, Y):
    '''Computes the normalised Hamming distance between two Bloom filter templates'''
    dist = 0

    N_BF = X.shape[0]
    for i in range(N_BF):
        A = X[i, :]
        B = Y[i, :]

        suma = sum(A) + sum(B)
        if suma > 0:
            dist += float(sum(A ^ B)) / float(suma)

    return dist / float(N_BF)

####################################################################
### Score computation

# read protocol files
matedF = open(matedComparisonsFile, 'r')
nonMatedF = open(nonMatedComparisonsFile, 'r')

# pre-allocate score arrays
matedScoresBF = []
nonMatedScoresBF = []

# compute scores for each reference template and save at each iteration
for l in matedF.readlines():
    r = l.split()

    aBF = numpy.loadtxt(DB_BFtemplates + r[0]).astype(int)
    bBF = numpy.loadtxt(DB_BFtemplates + r[1]).astype(int)
    matedScoresBF.append(hamming_distance(aBF, bBF))

for l in nonMatedF.readlines():
    r = l.split()

    aBF = numpy.loadtxt(DB_BFtemplates + r[0]).astype(int)
    bBF = numpy.loadtxt(DB_BFtemplates + r[1]).astype(int)
    nonMatedScoresBF.append(hamming_distance(aBF, bBF))

numpy.savetxt(scoresDir+matedScoresFile, matedScoresBF)
numpy.savetxt(scoresDir+nonMatedScoresFile, nonMatedScoresBF)
