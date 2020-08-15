library(knitr)
library(scatterplot3d)
library(Rtsne)
library(coRanking)


library(writexl)
library(readxl)

npoints <- 1000

theta <- runif(npoints, 0, 2 * pi)
u <- runif(npoints, -1, 0.8)

data <- list()
data$x <- sqrt(1 - u ^ 2) * cos(theta)
data$y <- sqrt(1 - u ^ 2) * sin(theta)
data$z <- u
data$col <-  rgb(colorRamp(colors = c("red", "yellow", "green"))( (data$z + 1) / 2),      maxColorValue = 255)
data <- as.data.frame(data, stringsAsFactors = F)

X_train_t=read_excel("G:/NCTUee/DataMining/OPC/SCLearn/DR/Rscript/X_train_t.xlsx")
X_train=read_excel("G:/NCTUee/DataMining/OPC/SCLearn/DR/Rscript/X_train.xlsx")

scatterplot3d(data$x, data$y, data$z,  xlab = "x", ylab = "y",zlab = "z")


Q.tsne <- coranking(X_train, X_train_t)

x=matrix(0, 474, 474)

for (ii in 1:474){
  for (jj in 1:474){
    x[ii,jj]=Q.tsne[ii,jj]
  }
  
}



imageplot(Q.tsne, main = "t-SNE")


lcmc.tsne <- numeric(nrow(Q.tsne))

lcmc.tsne <- LCMC(Q.tsne)

Kmax <- which.max(lcmc.tsne)




write_xlsx(as.data.frame(x), "G:/NCTUee/DataMining/OPC/SCLearn/DR/Rscript/mydata.xlsx")
write_xlsx(as.data.frame(lcmc.tsne), "G:/NCTUee/DataMining/OPC/SCLearn/DR/Rscript/mydata2.xlsx")

