# Data-Mining-class-competition

## Medical-Text-Classification

The goal of this competition is to allow to develop predictive models that can determine, given a particular medical abstract, which one of 5 classes it belongs to.

**Rank: 1/43**

### Dataset

- train.dat: 14442 records
Training set (class label, followed by a tab separating character and the text of the medical abstract).
- test.dat: 14438 records
Testing set (text of medical abstracts in lines, no class label provided).
- format.dat: A sample submission with 14438 entries randomly chosen to be 1 to 5.

## Image Classification

In this competition analyze features extracted from traffic images depicting different objects to determine their type as one of 11 classes, noted by integers 1-11: car, suv, small_truck, medium_truck, large_truck, pedestrian, bus, van, people, bicycle, and motorcycle. The object classes are heavily imbalanced.

#### Dataset
- 512 Histogram of Oriented Gradients (HOG) features - 256 Normalized Color Histogram (Hist) features
- 64 Local Binary Pattern (LBP) features
- 48 Color gradient (RGB) features
- 7 Depth of Field (DF) features