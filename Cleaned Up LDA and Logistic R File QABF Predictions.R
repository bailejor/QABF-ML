# 1 Loading relevant package and data ----
library(MASS)
QABFData <- read.csv("QABFData.csv")
QABFData <- QABFData[complete.cases(QABFData),]

# removes empty rows
rownames(QABFData) <- NULL

# variables to identify the relevant functions to remove redundancy later, based on columns in the original data
attention <- c(1,6,11)
escape <- attention+1
social <- escape+1
physical <- social+1
tangible <- physical+1
functionmatrix <- matrix(c(attention, escape, social, physical, tangible), ncol = 3, byrow = TRUE)
functionlist <- list(attention, escape, social, physical, tangible)

# convenient way to make formulas to use for LDA later
testing <- (paste(names(QABFData)[1:10], collapse = "+"))
testing2 <- paste("RealFunction1~", testing)
functionnames <- c("attention", "escape", "non-social", "physical", "tangible")

#Extra columns to store data later
QABFData$MaxFunction1 <- NA
QABFData$MaxFunction2 <- NA
QABFData$MaxFunction3 <- NA
QABFData$RealFunction1 <- NA
QABFData$RealFunction2 <- NA
QABFData$RealFunction3 <- NA

# Whole set is defined here
wholeset <- c(1:42)

# 2 Determines and assigns what the max scores are from the endorsement and intensity scores to columns 16 to 18 ----
for (a in 1:nrow(QABFData)){
    maxvector <- (as.numeric()) 
    for (b in 1:nrow(functionmatrix)){
        maxvector[b] <- QABFData[a, functionmatrix[b, 1]] + QABFData[a, functionmatrix[b, 2]]
    }
    maxfunc <- which(maxvector == max(maxvector))
    for(d in functionnames){
        if(length(maxfunc) == 3){
            QABFData[a, 16:18] <- functionnames[maxfunc]
        } else if(length(maxfunc) == 2){
            QABFData[a, 16:17] <- functionnames[maxfunc]
        } else if(length(maxfunc) == 1){
            QABFData[a, 16] <- functionnames[maxfunc]
        }
    }
}

# Assigns the randomly selected actual functions for use later
for (a in seq(QABFData[,1])){
    if (length(which(QABFData[a,11:15] == 1)) == 1) {
        QABFData[a, 19] <- functionnames[which(QABFData[a,11:15] == 1)]
    } else if (length(which(QABFData[a,11:15] == 1)) == 2){
        QABFData[a, 19:20] <- functionnames[sample(which(QABFData[a,11:15] == 1))]
    } else if (length(which(QABFData[a,11:15] == 1)) == 3){
        QABFData[a, 19:21] <- functionnames[sample(which(QABFData[a,11:15] == 1))]
    }
}

# Randomly samples different functions, and then compares the LDA predictions against ANY of the actual outcomes
OriginalLDARandomizer <- function(MM){ 
    results <- vector(length = MM)
    PlaceHeld <- QABFData
    WHOLETRUTH <- vector()
    for (B in 1:MM){
        for (a in seq(PlaceHeld[,1])){
            if (length(which(PlaceHeld[a,11:15] == 1)) == 1) {
                PlaceHeld[a, 19] <- functionnames[which(PlaceHeld[a,11:15] == 1)]
            } else if (length(which(PlaceHeld[a,11:15] == 1)) == 2){
                PlaceHeld[a, 19:20] <- functionnames[sample(which(PlaceHeld[a,11:15] == 1))]
            } else if (length(which(PlaceHeld[a,11:15] == 1)) == 3){
                PlaceHeld[a, 19:21] <- functionnames[sample(which(PlaceHeld[a,11:15] == 1))]
            }
        }
        LDAwhole <- lda(formula(testing2),data = PlaceHeld, CV = TRUE)
        for (m in 1:length(LDAwhole$class)){
            # print(LDAwhole$class[m] %in% PlaceHeld[m, 19:21]) ## Print results to console if you want
            WHOLETRUTH[m] <- LDAwhole$class[m] %in% PlaceHeld[m, 19:21]
        }
        results[B] <- mean(WHOLETRUTH)
        # print(results[B]) ## Print out final percentage assigned for each simulation
    }
    return(results)
}

# Output and descriptives for results 
LDAMultiClassStorage <- OriginalLDARandomizer(1000)
LDAMultiClassResults <- cbind(mean(LDAMultiClassStorage), 
      sd(LDAMultiClassStorage),
      max(LDAMultiClassStorage),
      min(LDAMultiClassStorage))
colnames(LDAMultiClassResults) <- c("Mean", "SD", "Max", "Min")
LDAMultiClassResults    

# 3 Used to create every model and LOOCV for endorsement score logistic regressions ---- 
quickmethodsLogisticSingleLOOCV <- function(){
    Q <- 1
    storagescoreLOOCV <- list()
    for(b in functionlist){
        storagescoreLOOCV[[Q]] <- list()
        for(a in wholeset){
            traindata <- QABFData[-a, b[c(1,3)]]
            Y <- paste0(names(traindata[,2, drop = FALSE]), "~", names(traindata[,1, drop = FALSE]))
            testingdata <- (QABFData[a, b[1], drop = FALSE])
            logsingle <- glm(data = QABFData, formula(Y), family = "binomial")
            # logsingle <- glm(data = QABFData, (names(traindata[, 2, drop = FALSE])) ~ as.formula(names(traindata[,1, drop = FALSE])), family = "binomial")
            predictions <- predict(logsingle, newdata = testingdata, type = "response")
            matrixresults <- data.frame(forumla = Y, set = a, 
                                        predictions = predict(logsingle, newdata = testingdata, type = "response"),
                                        roundedpredictions = round(predict(logsingle, newdata = testingdata, type = "response")),
                                        truth = QABFData[a, b[3]],
                                        accuracy =round(predict(logsingle, newdata = testingdata, type = "response")) == QABFData[a, b[3]])
            storagescoreLOOCV[[Q]][[a]] <- list(logsingle, predictions, matrixresults)
        }
        Q <-  Q + 1
    }
    return(storagescoreLOOCV)
}

tryinglists <- quickmethodsLogisticSingleLOOCV()

# creates a new list that contains all the predictions to use for comparison, endorsement 
truthings <- list()
for(a in 1:5){
    truthings[[a]] <- list() 
    for (b in 1:42) {
        truthings[[a]][b] <- tryinglists[[a]][[b]][[3]][6]
    }
}

# average LOOCV accuracy for endorsesment scores
averagesEndorsement <- vector(length=5)
for(avg in 1:5) averagesEndorsement[avg] <- (sum(unlist(truthings[[avg]]))/42)
names(averagesEndorsement) <- c(functionnames)
averagesEndorsement

# to create the overall accuracy matrix for endorsement scores
comparisonsEndorsement <- matrix(nrow = 42, ncol = 5)
for(something in 1:5) comparisonsEndorsement[, something] <- unlist(truthings[[something]])

logisticEndorsementOverallAcc <- mean(apply(comparisonsEndorsement, 1, all))
logisticEndorsementOverallAcc

# 4 intensity score logistic regressions and LOOCV ----
quickmethodsLogisticSingleINTLOOCV <- function(){
    Q <- 1
    M <- 0
    storageINTLOOCV <- list()
    for(b in functionlist){
        storageINTLOOCV[[Q]] <- list()
        for(a in wholeset){
            traindata <- QABFData[-a, b[c(2,3)]]
            Y <- paste0(names(traindata[,2, drop = FALSE]), "~", names(traindata[,1, drop = FALSE]))
            testingdata <- (QABFData[a, b[2], drop = FALSE])
            logsingle <- glm(data = QABFData, formula(Y), family = "binomial")
            # logsingle <- glm(data = QABFData, (names(traindata[, 2, drop  = FALSE])) ~ as.formula(names(traindata[,1, drop = FALSE])), family = "binomial")
            predictions <- predict(logsingle, newdata = testingdata, type = "response")
            matrixresults <- data.frame(forumla = Y, set = a, 
                                        predictions = predict(logsingle, newdata = testingdata, type = "response"),
                                        roundedpredictions = round(predict(logsingle, newdata = testingdata, type = "response")),
                                        truth = QABFData[a, b[3]],
                                        accuracy =round(predict(logsingle, newdata = testingdata, type = "response")) == QABFData[a, b[3]])
            storageINTLOOCV[[Q]][[a]] <- list(logsingle, predictions, matrixresults)
        }
        Q <-  Q + 1
    }
    return(storageINTLOOCV)
}

# creates a new list that contains all the predictions to use for comparison, intensity 
tryinglists2 <- quickmethodsLogisticSingleINTLOOCV()
truthings2 <- list()
for(a in 1:5){
    truthings2[[a]] <- list()
    for (b in 1:42) {
        # print(tryinglists2[[a]][[b]][[3]][6])
        truthings2[[a]][[b]] <- list(tryinglists2[[a]][[b]][[3]][6])
    }
}

# Intensity LOOCV accuracy 
averagesIntensity <- vector(length=5)
for(avg in 1:5) averagesIntensity[avg] <- (sum(unlist(truthings2[[avg]]))/42)
names(averagesIntensity) <- c(functionnames)
averagesIntensity

# Matrix from the intensity to create overall accuracy 
comparisonsIntensity <- matrix(nrow = 42, ncol = 5)
for(something in 1:5) comparisonsIntensity[, something] <- unlist(truthings2[[something]])
comparisonsIntensity
logisticIntensityOverallAcc <- mean(apply(comparisonsIntensity, 1, all))
logisticIntensityOverallAcc

# 5 Logistic regressions with endorsement and intensity ----
quickmethodsLogisticCombinedLOOCV <- function(){
    Q <- 1
    M <- 0
    storageComLOOCV <- list()
    for(b in functionlist){
        storageComLOOCV[[Q]] <- list()
        for(a in wholeset){
            traindata <- QABFData[-a, b[c(1:3)]]
            Y <- paste0(names(traindata[,3, drop = FALSE]), "~", names(traindata[,1, drop = FALSE]), "+",names(traindata[,2, drop = FALSE]) )
            testingdata <- (QABFData[a, b[1:2], drop = FALSE])
            logsingle <- glm(data = QABFData, formula(Y), family = "binomial")
            # logsingle <- glm(data = QABFData, (names(traindata[, 2, drop = FALSE])) ~ as.formula(names(traindata[,1, drop = FALSE])), family = "binomial")
            predictions <- predict(logsingle, newdata = testingdata, type = "response")
            matrixresults <- data.frame(forumla = Y, set = a, 
                                        predictions = predict(logsingle, newdata = testingdata, type = "response"),
                                        roundedpredictions = round(predict(logsingle, newdata = testingdata, type = "response")),
                                        truth = QABFData[a, b[3]],
                                        accuracy =round(predict(logsingle, newdata = testingdata, type = "response")) == QABFData[a, b[3]])
            storageComLOOCV[[Q]][[a]] <- list(logsingle, predictions, matrixresults)
        }
        Q <-  Q + 1
    }
    return(storageComLOOCV)
}

tryinglists3 <- quickmethodsLogisticCombinedLOOCV()
truthings3 <- list()
for(a in 1:5){
    truthings3[[a]] <- list()
    for (b in 1:42) {
        # print(tryinglists3[[a]][[b]][[3]][6])
        truthings3[[a]][[b]] <- list(tryinglists3[[a]][[b]][[3]][6])
    }
}

# Combined LOOCV endorsement and intensity accuracy 
averagesCombined <- vector(length=5)
for(avg in 1:5) averagesCombined[avg] <- (sum(unlist(truthings3[[avg]]))/42)
names(averagesCombined) <- c(functionnames)
averagesCombined

# Overall accuracy for combined intesity and endorsement scores
comparisonsCombined <- matrix(nrow = 42, ncol = 5)
for(something in 1:5) comparisonsCombined[, something] <- unlist(truthings3[[something]])
comparisonsCombined
logisticCombinedOverallAcc <- mean(apply(comparisonsCombined, 1, all))
logisticCombinedOverallAcc

# 6 All predictors in logistic model ----
# Function to cycle through and create formulas for models to be used for all predictors and LOOCV
magicfunction <- function(L){
    predictvector <- as.numeric()
    for(a in wholeset){
        # print(a)
        trainingDATA <- QABFData[-a, c(1:10, L)]
        testingDATA <- QABFData[a, c(1:10, L)]
        colnames(testingDATA)[11] <- "Y"
        colnames(trainingDATA)[11] <- "Y"
        model1 <- glm(formula(Y ~ .), family = "binomial", data = trainingDATA)
        predictvector[a] <- round(predict(model1, testingDATA, type = "response"))
    }
    return(predictvector)
}

# combines all the above to create a LOOCV for each function with all predictors inside
allfunctionloop <- function(){
    allvector <- data.frame(Attention = rep(NA, 42), Escape = rep(NA, 42), Non.Social = rep(NA, 42), Physical = rep(NA, 42), Tangible = rep(NA, 42)) 
    # print(allvector)
    for(L in 11:15){
        # print(L)
        allvector[,(L-10)] <- magicfunction(L)
    }
    return(allvector)
}
allresults <- allfunctionloop()
truthall <- allresults == QABFData[,11:15]

# results for all predictors in the logistic regressions
BinaryAllPredictorsAccuracy <- apply(truthall, 2, mean)
BinaryAllPredictorsAccuracy
OverallAllPredictorsAccuracy <- mean(apply(truthall, 1, all))
OverallAllPredictorsAccuracy

# 7 LDA binary prediction accuracy vectors ----
LDATRUTH <- cbind((lda(formula = Att...FA ~ ., data = QABFData[,c(1:11)], CV = TRUE)$class == QABFData[,11]),  
                  (lda(formula = Esc...FA ~ ., data = QABFData[,c(1:10,12)], CV = TRUE)$class == QABFData[,12]),
                  (lda(formula = Non.soc....FA ~ ., data = QABFData[,c(1:10,13)], CV = TRUE)$class == QABFData[,13]),  
                  (lda(formula = Phys...FA ~ ., data = QABFData[,c(1:10,14)], CV = TRUE)$class == QABFData[,14]),  
                  (lda(formula = Tang...FA ~ ., data = QABFData[,c(1:10,15)], CV = TRUE)$class == QABFData[,15]))  
LDATHEORTICALPREDICTORS <- cbind((lda(formula = Att...FA ~ ., data = QABFData[,c(1,2,11)], CV = TRUE)$class == QABFData[,11]),  
                                 (lda(formula = Esc...FA ~ ., data = QABFData[,c(3,4,12)], CV = TRUE)$class == QABFData[,12]),
                                 (lda(formula = Non.soc....FA ~ ., data = QABFData[,c(5,6,13)], CV = TRUE)$class == QABFData[,13]),  
                                 (lda(formula = Phys...FA ~ ., data = QABFData[,c(7,8,14)], CV = TRUE)$class == QABFData[,14]),  
                                 (lda(formula = Tang...FA ~ ., data = QABFData[,c(9,10,15)], CV = TRUE)$class == QABFData[,15]))  
# removes NAs
LDATRUTH[is.na(LDATRUTH)] <- FALSE

# Overall accuracy of LDA brinary predictions 
OverallPredictionLDAAccuracy <- mean(apply(LDATRUTH, 1, all))
OverallPredictionLDAAccuracy
binaryOverallPredicition <- apply(LDATRUTH, 2, mean)
names(binaryOverallPredicition) <- functionnames
binaryOverallPredicition

# Binary LDA with only theoretically relevant endorsements and intensity scores 
OverallTheoreticalLDAAccuracy <- mean(apply(LDATHEORTICALPREDICTORS, 1, all))
OverallTheoreticalLDAAccuracy
theoreticalLDAAccuracy <- apply(LDATHEORTICALPREDICTORS, 2, mean)
names(theoreticalLDAAccuracy) <- functionnames
theoreticalLDAAccuracy

# 8 Logistic Regressions Full Models ----
# endorsement scores
attglmscore <- glm(data = QABFData[, attention], Att...FA ~ Attention, family = "binomial")
escapeglmscore <- glm(data = QABFData[, escape], Esc...FA ~ Escape, family = "binomial")
socialglmscore <- glm(data = QABFData[, social], Non.soc....FA ~ Non.social, family = "binomial")
physicalglmscore <- glm(data = QABFData[, physical], Phys...FA ~ Physical, family = "binomial")
tangibleglmscore <- glm(data = QABFData[, tangible], Tang...FA ~ Tangible, family = "binomial")

# intensity scores
attglmint <- glm(data = QABFData[, attention], Att...FA ~ Att...Intensity, family = "binomial")
escapeglmint <- glm(data = QABFData[, escape], Esc...FA ~ Esc...Intensity, family = "binomial")
socialglmint <- glm(data = QABFData[, social], Non.soc....FA ~ Non.soc...Intensity, family = "binomial")
physicalglmint <- glm(data = QABFData[, physical], Phys...FA ~ Phys...Intensity, family = "binomial")
tangibleglmint <- glm(data = QABFData[, tangible], Tang...FA ~ Tang...Intensity, family = "binomial")

# combination tests 
# theoretical predictors
attglmcombined <- glm(data = QABFData[, attention], Att...FA ~ Attention + Att...Intensity, family = "binomial")
escapeglmcombined <- glm(data = QABFData[, escape], Esc...FA ~ Escape + Esc...Intensity, family = "binomial")
socialglmcombined <- glm(data = QABFData[, social], Non.soc....FA ~ Non.social + Non.soc...Intensity, family = "binomial")
physicalglmcombined <- glm(data = QABFData[, physical], Phys...FA ~ Physical + Phys...Intensity, family = "binomial")
tangibleglmcombined <- glm(data = QABFData[, tangible], Tang...FA ~ Tangible + Tang...Intensity, family = "binomial")

# all predictors
attglmAllVars <- glm(data = QABFData[, c(1:10, 11)], Att...FA ~ ., family = "binomial")
escapeglmAllVars <- glm(data = QABFData[, c(1:10, 12)], Esc...FA ~ ., family = "binomial")
socialglmAllVars <- glm(data = QABFData[, c(1:10, 13)], Non.soc....FA ~ ., family = "binomial")
physicalglmAllVars <- glm(data = QABFData[, c(1:10, 14)], Phys...FA ~ ., family = "binomial")
tangibleglmAllVars <- glm(data = QABFData[, c(1:10, 15)], Tang...FA ~ ., family = "binomial")
