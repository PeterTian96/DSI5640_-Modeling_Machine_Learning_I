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

# Question 2: Using the mcycle data, consider predicting the mean acceleration as a function of time. Use the Nadaraya-Watson method with the k-NN kernel

# Question 3: function to create a series of prediction models by varying the tuning parameter over a sequence of values. (hint: the script already implements this)

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

# Question 4: With the squared-error loss function, compute and plot the training error, AIC, BIC, and validation error (using the validation data) as functions of the tuning parameter.

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
# manipulate({
#   ## make predictions using NW method at training inputs
#   y_hat <- nadaraya_watson(y, x, x,
#     kern=kernel_k_nearest_neighbors, k=k_slider)
#   edf <- effective_df(y, x, 
#     kern=kernel_k_nearest_neighbors, k=k_slider)
#   aic_ <- aic(y, y_hat, edf)
#   bic_ <- bic(y, y_hat, edf)
#   y_hat_plot <- nadaraya_watson(y, x, x_plot,
#     kern=kernel_k_nearest_neighbors, k=k_slider)
#   plot(x, y, xlab="Time (ms)", ylab="Acceleration (g)")
#   legend('topright', legend = c(
#     paste0('eff. df = ', round(edf,1)),
#     paste0('aic = ', round(aic_, 1)),
#     paste0('bic = ', round(bic_, 1))),
#     bty='n')
#   lines(x_plot, y_hat_plot, col="#882255", lwd=2) 
# }, k_slider=slider(1, 10, initial=3, step=1))
```

# Question 5: For each value of the tuning parameter, Perform 5-fold cross-validation using the combined training and validation data. This results in 5 estimates of test error per tuning parameter value.

## 5-fold cross-validation of knnreg model

``` r
## create five folds
set.seed(1)
mcycle_flds  <- createFolds(mcycle$accel, k=5)
print(mcycle_flds)
```

    ## $Fold1
    ##  [1]   2   5  11  12  15  23  26  45  46  51  54  58  66  75  76  80  83  84  87
    ## [20]  90  96  98 115 122 128 131 132
    ## 
    ## $Fold2
    ##  [1]   8  14  17  25  30  33  36  41  42  43  44  48  49  68  78  81  91  93  94
    ## [20]  99 102 106 108 110 126 127 130
    ## 
    ## $Fold3
    ##  [1]   4   7  21  22  24  28  29  39  47  50  53  55  61  64  67  69  79  86  92
    ## [20] 105 107 112 120 121 123 133
    ## 
    ## $Fold4
    ##  [1]   1   3   6   9  16  18  31  32  34  37  40  52  57  59  63  70  88  97 100
    ## [20] 101 103 114 116 118 124 129
    ## 
    ## $Fold5
    ##  [1]  10  13  19  20  27  35  38  56  60  62  65  71  72  73  74  77  82  85  89
    ## [20]  95 104 109 111 113 117 119 125

``` r
sapply(mcycle_flds, length) 
```

    ## Fold1 Fold2 Fold3 Fold4 Fold5 
    ##    27    27    26    26    27

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

# Question 6: Plot the CV-estimated test error (average of the five estimates from each fold) as a function of the tuning parameter. Add vertical line segments to the figure (using the segments function in R) that represent one “standard error” of the CV-estimated test error (standard deviation of the five estimates from each fold).

# CV-estimated test error

``` r
## Compute 5-fold CV for kNN = 1:20
cverrs <- sapply(1:20, cvknnreg)
print(cverrs) ## rows are k-folds (1:5), cols are kNN (1:20)
```

    ##           [,1]      [,2]      [,3]      [,4]      [,5]      [,6]      [,7]
    ## [1,]  833.7631  582.3799  520.4267  524.4643  541.7711  478.6416  445.9896
    ## [2,] 1352.1064 1381.4486 1297.2430 1324.4854 1098.2313 1122.7449 1141.9824
    ## [3,]  984.5000  632.1399  456.9012  488.1716  557.1328  522.9114  539.9713
    ## [4,] 1177.9617  893.0425  631.6250  648.2299  643.6340  697.5926  662.9493
    ## [5,]  594.9439  573.1915  439.9326  395.8718  415.9289  400.8986  348.2554
    ##           [,8]      [,9]     [,10]     [,11]     [,12]     [,13]    [,14]
    ## [1,]  398.2609  384.7866  407.2981  446.4139  434.4257  428.8328 447.7663
    ## [2,] 1106.4601 1100.6343 1008.7705 1057.8712 1039.9583 1003.2420 961.8530
    ## [3,]  536.8323  580.3196  613.0110  624.9374  628.4059  645.9322 656.1005
    ## [4,]  619.5004  605.5366  593.4215  576.6957  606.5749  610.0195 587.4333
    ## [5,]  372.7873  459.0067  489.4587  508.5917  470.9600  418.4527 461.5154
    ##         [,15]    [,16]    [,17]    [,18]    [,19]    [,20]
    ## [1,] 418.1714 413.5000 416.8825 408.8881 434.2074 426.0638
    ## [2,] 972.7536 951.6233 949.5257 976.1762 998.5977 962.6726
    ## [3,] 670.7562 698.0429 718.1851 784.7335 817.6874 859.8918
    ## [4,] 593.5669 562.1299 568.0380 611.9726 624.0383 662.2680
    ## [5,] 518.0408 519.1216 511.8726 544.0493 560.4584 623.2298

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

![](Homework5_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

# Question 7:Interpret the resulting figures and select a suitable value for the tuning parameter.

Because of the 1sd rule, Here the k = 23 will be the best one