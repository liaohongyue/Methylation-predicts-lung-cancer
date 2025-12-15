library("limma")
library("dplyr")
args<-commandArgs(T)
sample<-args[1]
# sample_name<-args[2]
n1=args[2]
n2=args[3]
n3=args[4]

meth1 <- read.table(sample,header=T,sep=",",row.names=1,comment.char="",check.names=F)
meth <- meth1[!rownames(meth1)=="sample_type",]

epsilon <-1e-6
meth <- log2(meth+epsilon)

list <- c(rep("Treat", n1),rep("CK",n2)) %>% factor(., levels = c("CK", "Treat"), ordered = F)
list <- model.matrix(~factor(list)+0)#把group设置成一个model matrix

colnames(list) <- c("Treat", "CK")
meth.fit <- lmFit(meth, list)
meth.matrix <- makeContrasts(Treat - CK, levels = list)
fit <- contrasts.fit(meth.fit, meth.matrix)
fit <- eBayes(fit)
tempOutput <- topTable(fit,n = Inf,coef=1,adjust = "fdr")
nrDEG <- na.omit(tempOutput)
diffsig <- nrDEG
diffsig <- diffsig[diffsig$adj.P.Val<1,]
diffsi <- diffsig[order(diffsig$logFC),]
file_name <- paste(n3,sep="")

write.table(data.frame("chrBase"=rownames(diffsi),diffsi,check.names = FALSE),file=file_name,sep='\t',row.names=F)
