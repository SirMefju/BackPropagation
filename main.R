set.seed(2137)
data <- read.csv("iris.csv") # importowanie danych
data <- data[sample(nrow(data)), ] #losowanie rzedow
dataTrain <- 0.8 * nrow(data) # podział danych
train <- data[1:dataTrain, ] # dane treningowe
test <- data[(dataTrain+1):nrow(data), ] # dane testowe
xTrain <- scale(train[, (1:2)]) # standaryzacja danych (scale)
yTrain <- train$y # odwolanie sie do wyjscia algorytmu
dim(yTrain) <- c(length(yTrain),1) # wymiar wektora
xTest <- scale(test[, (1:2)]) # standaryzacja danych (scale)
yTest <- test$y # odwolanie sie do wyjscia algorytmu
dim(yTest) <- c(length(yTest),1) # wymiar wektora
xTrain <- as.matrix(xTrain, byrow=TRUE) # wypalnianie macierzy wiersz po wierszu
xTrain <- t(xTrain) # transponacja macierzy zeby latwiej liczyc
yTrain <- as.matrix(yTrain, byrow=TRUE)
yTrain <- t(yTrain)
xTest <- as.matrix(xTest, byrow=TRUE)
xTest <- t(xTest)
yTest <- as.matrix(yTest, byrow=TRUE)
yTest <- t(yTest)
getLayerSize <- function(x, y,hiddenNeurons, train=TRUE){
  nx <- dim(x)[1] # wymiar warstwy wejsciowej
  nh <- hiddenNeurons # wymiar warstwy ukrytej
  ny <- dim(y)[1] # wymiar warstwy wyjsciowej
  size <- list("nx" = nx,
               "nh" = nh,
               "ny" = ny)
  return(size)
  }
layerSize <- getLayerSize(xTrain, yTrain, hiddenNeurons = 2)
initializeParameters <- function(x, listLayerSize){
  m <- dim(data.matrix(x))[4]
  nx <- listLayerSize$nx
  nh <- listLayerSize$nh
  ny <- listLayerSize$ny
  w1 <- matrix(runif(nh * nx), nrow = nh, ncol = nx, byrow = TRUE) * 0.01
  b1 <- matrix(rep(0, nh), nrow = nh)
  w2 <- matrix(runif(ny * nh), nrow = ny, ncol = nh, byrow = TRUE) * 0.01
  b2 <- matrix(rep(0, ny), nrow = ny)
  parameters <- list("w1" = w1, "b1" = b1, "w2" = w2, "b2" = b2)
  return (parameters)
}
initializationParameters <- initializeParameters(xTrain, layerSize)
lapply(initializationParameters, function(x) dim(x))
sigmoid <- function(x){
    return(1 / (1 + exp(-x)))
} # funkcja aktywacji (sigmoidalna)
forwardPropagation <- function(x, parameters, listLayerSize){
  m <- dim(x)[2]
  nh <- listLayerSize$nh
  ny <- listLayerSize$ny
  w1 <- parameters$w1
  b1 <- parameters$b1
  w2 <- parameters$w2
  b2 <- parameters$b2
  b1New <- matrix(rep(b1, m), nrow = nh)
  b2New <- matrix(rep(b2, m), nrow = ny)
  z1 <- w1 %*% x + b1New
  a1 <- sigmoid(z1)
  z2 <- w2 %*% a1 + b2New
  a2 <- sigmoid(z2)
  cache <- list("z1" = z1,
                "a1" = a1,
                "z2" = z2,
                "a2" = a2)
  return (cache)
}
forwardPropagationx <- forwardPropagation(xTrain, initializationParameters, layerSize)
lapply(forwardPropagationx, function(x) dim(x)) # dane jako lista
computeCost <- function(x, y, cache) {
    m <- dim(x)[2]
    a2 <- cache$a2
    logprobs <- (log(a2) * y) + (log(1-a2) * (1-y))
    cost <- -sum(logprobs/m)
    return (cost)
}
cost <- computeCost(xTrain, yTrain, forwardPropagationx)
cost
backwardPropagation <- function(x, y, cache, parameters, listLayerSize){
    m <- dim(x)[2]
    nx <- listLayerSize$nx
    nh <- listLayerSize$nh
    ny <- listLayerSize$ny
    a2 <- cache$a2
    a1 <- cache$a1
    w2 <- parameters$w2
    dz2 <- a2 - y
    dw2 <- 1/m * (dz2 %*% t(a1))
    db2 <- matrix(1/m * sum(dz2), nrow = ny)
    db2New <- matrix(rep(db2, m), nrow = ny)
    dz1 <- (t(w2) %*% dz2) * (1 - a1^2)
    dw1 <- 1/m * (dz1 %*% t(x))
    db1 <- matrix(1/m * sum(dz1), nrow = nh)
    db1New <- matrix(rep(db1, m), nrow = nh)
    grads <- list("dw1" = dw1,
                  "db1" = db1,
                  "dw2" = dw2,
                  "db2" = db2)
    return(grads)
}
updateParameters <- function(grads, parameters, learningRate){
    w1 <- parameters$w1
    b1 <- parameters$b1
    w2 <- parameters$w2
    b2 <- parameters$b2
    dw1 <- grads$dw1
    db1 <- grads$db1
    dw2 <- grads$dw2
    db2 <- grads$db2
    w1 <- w1 - learningRate * dw1
    b1 <- b1 - learningRate * db1
    w2 <- w2 - learningRate * dw2
    b2 <- b2 - learningRate * db2
    updatedParameters <- list("w1" = w1,
                           "b1" = b1,
                           "w2" = w2,
                           "b2" = b2)
    return (updatedParameters)
}
trainModel <- function(x, y, numerIteration, hiddenNeurons, lr){
    layerSize <- getLayerSize(x, y, hiddenNeurons)
    initializeParameters <- initializeParameters(x, layerSize)
    costHistory <- c()
    iters <- c()
    for (i in 1:numerIteration) {
        forwardPropagation <- forwardPropagation(x, initializeParameters, layerSize)
        cost <- computeCost(x, y, forwardPropagation)
        backwardPropagation <- backwardPropagation(x, y, forwardPropagation, initializeParameters, layerSize)
        updateParameters <- updateParameters(backwardPropagation, initializeParameters, learningRate = lr)
        initializeParameters <- updateParameters
        costHistory <- c(costHistory, cost)
        iters <- c(iters, i)
        if (i %% 100 == 0) cat("Iteration", i, " | Cost: ", cost, "\n")
    }
    plot(iters, costHistory)
    modelOut <- list("updateParameters" = updateParameters,
                      "costHistory" = costHistory)
    return (modelOut)
}

##
Epochs = 60000
Hidden_Neurons = 40
Learning_Rate = 0.9
##

trainModel <- trainModel(xTrain, yTrain, hiddenNeurons = Hidden_Neurons, numerIteration = Epochs, lr = Learning_Rate)
makePrediction <- function(x, y, hiddenNeurons){
    layerSize <- getLayerSize(x, y, hiddenNeurons)
    parameters <- trainModel$updateParameters
    forwardPropagation <- forwardPropagation(x, parameters, layerSize)
    prediction <- forwardPropagation$a2
    return (prediction)
}
yPrediction <- makePrediction(xTest, yTest, Hidden_Neurons)
yPrediction <- round(yPrediction)
tb <- table(yTest, yPrediction)
cat("Confusion Matrix: \n")
tb
calculateStats <- function(tb, modelName) {
  acc <- (tb[1] + tb[4])/(tb[1] + tb[2] + tb[3] + tb[4])
  cat(modelName, "\n")
  cat("\tAccuracy = ", acc*100, "%.")
}
calcStats <- calculateStats(tb, "Irises:")
