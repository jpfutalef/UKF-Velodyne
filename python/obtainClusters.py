from personDetectionVelodyne import *
import sys

# para formar clusters raw
bagName = 'bagFile'

if len(sys.argv) > 1:
    bagName = sys.argv[1]

clustersToCSV(bagName)
