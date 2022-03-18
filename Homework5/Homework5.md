homework5
================

In the ISLR book, read section 6.1.3 “Choosing the Optimal Model” and
section 5.1 “Cross-Validation”. Extend and convert the attached
effective-df-aic-bic-mcycle.R R script into an R markdown file that
accomplishes the following tasks.

``` r
library('MASS') ## for 'mcycle'
library('manipulate') ## for 'manipulate'
library(tidyverse)
```

    ## ── Attaching packages ─────────────────────────────────────── tidyverse 1.3.1 ──

    ## ✓ ggplot2 3.3.5     ✓ purrr   0.3.4
    ## ✓ tibble  3.1.6     ✓ dplyr   1.0.7
    ## ✓ tidyr   1.1.4     ✓ stringr 1.4.0
    ## ✓ readr   2.0.2     ✓ forcats 0.5.1

    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## x dplyr::filter() masks stats::filter()
    ## x dplyr::lag()    masks stats::lag()
    ## x dplyr::select() masks MASS::select()

``` r
library(caret)
```

    ## Loading required package: lattice

    ## 
    ## Attaching package: 'caret'

    ## The following object is masked from 'package:purrr':
    ## 
    ##     lift

# Question 1: Randomly split the mcycle data into training (75%) and validation (25%) subsets.

``` r
#load data
mcycle <- mcycle
#split the mcycle data into training (75%) subsets
train<-sample_frac(mcycle, 0.75)
sid<-as.numeric(rownames(train))
# split the mcycle data into validation (25%) subsets
test<-mcycle[-sid,]
```

# Question 2: Using the mcycle data, consider predicting the mean acceleration as a function of time. Use the Nadaraya-Watson method with the k-NN kernel function to create a series of prediction models by varying the tuning parameter over a sequence of values. (hint: the script already implements this)

``` r
# predicting the mean acceleration
y <- train$accel
x <- matrix(train$times, length(train$times), 1)
```

## k-NN kernel function

``` r
## x  - n x p matrix of training inputs
## x0 - 1 x p input where to make prediction
## k  - number of nearest neighbors
kernel_k_nearest_neighbors <- function(x, x0, k=1) {
  ## compute distance betwen each x and x0
  z <- t(t(x) - x0)
  d <- sqrt(rowSums(z*z))

  ## initialize kernel weights to zero
  w <- rep(0, length(d))
  
  ## set weight to 1 for k nearest neighbors
  w[order(d)[1:k]] <- 1
  
  return(w)
}
```

## Make predictions using the NW method

``` r
## y  - n x 1 vector of training outputs
## x  - n x p matrix of training inputs
## x0 - m x p matrix where to make predictions
## kern  - kernel function to use
## ... - arguments to pass to kernel function
nadaraya_watson <- function(y, x, x0, kern, ...) {
  k <- t(apply(x0, 1, function(x0_) {
    k_ <- kern(x, x0_, ...)
    k_/sum(k_)
  }))
  yhat <- drop(k %*% y)
  attr(yhat, 'k') <- k
  return(yhat)
}
```

``` r
## create a grid of inputs 
x_plot <- matrix(seq(min(x),max(x),length.out=100),100,1)

## make predictions using NW method at training inputs
y_hat_plot <- nadaraya_watson(y, x, x_plot,
  kernel_k_nearest_neighbors)

## plot predictions
plot(x, y, xlab="Time (ms)", ylab="Acceleration (g)")
lines(x_plot, y_hat_plot, col="#882255", lwd=2) 
```

![](Homework5_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
## set k from 1 to 10
k_seq=seq(1, 10, by = 1)
## make predictions using NW method at training inputs
for (n in k_seq){
  y_hat_plot <- nadaraya_watson(y, x, x_plot, kernel_k_nearest_neighbors, n)
}
```

# Question 3: With the squared-error loss function, compute and plot the training error, AIC, BIC, and validation error (using the validation data) as functions of the tuning parameter.

``` r
#Use validation data
y <- test$accel
x <- matrix(test$times, length(test$times), 1)
```

## the squared-error loss function

``` r
## loss function
## y    - train/test y
## yhat - predictions at train/test x
loss_squared_error <- function(y, yhat)
  (y - yhat)^2
```

## test/train error

``` r
## y    - train/test y
## yhat - predictions at train/test x
## loss - loss function
error <- function(y, yhat, loss=loss_squared_error)
  mean(loss(y, yhat))
```

## AIC

``` r
## y    - training y
## yhat - predictions at training x
## d    - effective degrees of freedom
aic <- function(y, yhat, d)
  error(y, yhat) + 2/length(y)*d
```

## BIC

``` r
## y    - training y
## yhat - predictions at training x
## d    - effective degrees of freedom
bic <- function(y, yhat, d)
  error(y, yhat) + log(length(y))/length(y)*d
```

``` r
## Compute effective df using NW method
## y  - n x 1 vector of training outputs
## x  - n x p matrix of training inputs
## kern  - kernel function to use
## ... - arguments to pass to kernel function
effective_df <- function(y, x, kern, ...) {
  y_hat <- nadaraya_watson(y, x, x,
    kern=kern, ...)
  sum(diag(attr(y_hat, 'k')))
}
```

``` r
## how does k affect shape of predictor and eff. df using k-nn kernel ?

   ## make predictions using NW method at training inputs
   y_hat <- nadaraya_watson(y, x, x,
     kern=kernel_k_nearest_neighbors, k=1)
   edf <- effective_df(y, x, 
     kern=kernel_k_nearest_neighbors, k=1)
   aic_ <- aic(y, y_hat, edf)
   bic_ <- bic(y, y_hat, edf)
   y_hat_plot <- nadaraya_watson(y, x, x_plot,
     kern=kernel_k_nearest_neighbors, k=1)
   plot(x, y, xlab="Time (ms)", ylab="Acceleration (g)")
   legend('topright', legend = c(
     paste0('eff. df = ', round(edf,1)),
     paste0('aic = ', round(aic_, 1)),
     paste0('bic = ', round(bic_, 1))),
     bty='n')
   lines(x_plot, y_hat_plot, col="#882255", lwd=2) 
```

![](Homework5_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

# Question 4: For each value of the tuning parameter, Perform 5-fold cross-validation using the combined training and validation data. This results in 5 estimates of test error per tuning parameter value.

## 5-fold cross-validation of knnreg model

``` r
## create five folds
set.seed(10)
mcycle_flds  <- createFolds(mcycle$accel, k=5)
print(mcycle_flds)
```

    ## $Fold1
    ##  [1]  12  13  19  21  26  29  40  56  58  59  68  73  74  79  80  96 107 109 113
    ## [20] 117 118 119 121 123 124 128
    ## 
    ## $Fold2
    ##  [1]   3   5   6  14  25  33  48  49  53  54  66  67  71  78  81  88  91 103 104
    ## [20] 112 114 120 122 125 126 127
    ## 
    ## $Fold3
    ##  [1]   4  10  15  16  23  24  31  36  39  42  43  47  52  57  61  64  69  76  83
    ## [20]  85  87  90  92  93  97 110 116 132
    ## 
    ## $Fold4
    ##  [1]   1   7   8  17  22  34  35  38  41  46  50  51  60  63  70  82  89  98 100
    ## [20] 101 105 106 108 111 115 130 131
    ## 
    ## $Fold5
    ##  [1]   2   9  11  18  20  27  28  30  32  37  44  45  55  62  65  72  75  77  84
    ## [20]  86  94  95  99 102 129 133

``` r
sapply(mcycle_flds, length) 
```

    ## Fold1 Fold2 Fold3 Fold4 Fold5 
    ##    26    26    28    27    26

## 5 estimates of test error

``` r
cvknnreg <- function(kNN = 10, flds=mcycle_flds) {
  cverr <- rep(NA, length(flds))
  for(tst_idx in 1:length(flds)) { ## for each fold
    
    ## get training and testing data
    mcycle_trn <- mcycle[-flds[[tst_idx]],]
    mcycle_tst <- mcycle[ flds[[tst_idx]],]
    
    ## fit kNN model to training data
    knn_fit <- knnreg(accel ~ times,
                      k=kNN, data=mcycle_trn)
    
    ## compute test error on testing data
    pre_tst <- predict(knn_fit, mcycle_tst)
    cverr[tst_idx] <- mean((mcycle_tst$accel - pre_tst)^2)
  }
  return(cverr)
}
```

# Question 5: Plot the CV-estimated test error (average of the five estimates from each fold) as a function of the tuning parameter. Add vertical line segments to the figure (using the segments function in R) that represent one “standard error” of the CV-estimated test error (standard deviation of the five estimates from each fold).

# CV-estimated test error

``` r
## Compute 5-fold CV for kNN = 1:20
cverrs <- sapply(1:20, cvknnreg)
print(cverrs) ## rows are k-folds (1:5), cols are kNN (1:20)
```

    ##           [,1]      [,2]     [,3]     [,4]     [,5]     [,6]     [,7]     [,8]
    ## [1,]  944.6754  324.2815 302.7030 320.2186 299.7381 274.1784 263.4059 286.1211
    ## [2,] 1208.9446  988.5358 994.1891 762.1013 819.1187 782.8219 792.4407 762.3584
    ## [3,]  730.9485  680.3268 557.4403 498.5075 438.9633 437.6091 429.0479 433.0094
    ## [4,] 1174.1995 1188.9012 840.4193 780.8227 694.7070 687.8277 669.2867 630.8863
    ## [5,]  820.2002  914.0428 716.4009 707.7038 742.3235 678.3217 702.1627 674.3803
    ##          [,9]    [,10]    [,11]    [,12]    [,13]    [,14]    [,15]    [,16]
    ## [1,] 344.5536 349.8934 330.6941 322.0038 336.7929 324.8618 332.6750 378.9130
    ## [2,] 746.9166 695.5230 695.6566 708.5470 700.7166 722.2612 740.2942 746.7839
    ## [3,] 417.3506 432.9561 385.7447 376.9452 386.6568 448.3049 469.1322 470.6407
    ## [4,] 666.6104 713.6058 716.5916 741.0025 763.4235 732.7333 768.6781 746.4518
    ## [5,] 671.2930 667.2644 651.0957 712.1556 695.8658 716.0543 723.9858 717.7456
    ##         [,17]    [,18]    [,19]    [,20]
    ## [1,] 394.6421 387.0614 431.6407 421.9411
    ## [2,] 813.4534 844.3363 842.5568 829.2088
    ## [3,] 493.2770 505.4170 504.7318 545.8710
    ## [4,] 774.7347 791.8819 746.8946 761.7921
    ## [5,] 690.5043 690.9477 755.7329 754.4012

``` r
cverrs_mean <- apply(cverrs, 2, mean)
cverrs_sd   <- apply(cverrs, 2, sd)
## Plot the results of 5-fold CV for kNN = 1:20
plot(x=1:20, y=cverrs_mean, 
     ylim=range(cverrs),
     xlab="'k' in kNN", ylab="CV Estimate of Test Error")
segments(x0=1:20, x1=1:20,
         y0=cverrs_mean-cverrs_sd,
         y1=cverrs_mean+cverrs_sd)
best_idx <- which.min(cverrs_mean)
points(x=best_idx, y=cverrs_mean[best_idx], pch=20)
abline(h=cverrs_mean[best_idx] + cverrs_sd[best_idx], lty=3) 
```

![](Homework5_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

# Question 6:Interpret the resulting figures and select a suitable value for the tuning parameter.

From the graph, the error goes down at the first and becomes higher
after K=11.the highest k value should be around 1 standard deviation
from k = 20.
