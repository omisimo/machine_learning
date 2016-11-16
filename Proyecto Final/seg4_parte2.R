

################################################################################
# SEGMENTO 4
################################################################################
#Replicando mejor modelo
seg4_new$preds<-NULL

seg4_new$llavepu<-seg4_llaves_new

bestiter_s4<-102
set.seed(bestiter_s4)
base<-seg4_new

malos<-subset(base,inc==1)   #Subset con clientes con incumplimiento igual a 1
num_malos<-nrow(malos)
buenos<-subset(base,inc==0) 

ind_samp<-sample(1:nrow(buenos), size=floor(num_malos/4), replace = FALSE, prob = NULL)
ind_malos<-sample(1:nrow(malos),size=floor(num_malos/4),replace=FALSE,prob=NULL)

samp<-buenos[ind_samp,]
samp_malos<-malos[ind_malos,]
samp_complemento<-buenos[-ind_samp,]
samp_malos_complemento<-malos[-ind_malos,]
base_complemento<-rbind(samp_complemento,samp_malos_complemento)

basesamp<-rbind(samp,samp_malos)
sub.base<-basesamp
sub.base$inc<-as.factor(sub.base$inc)

seg4_llaves_train<-sub.base$llavepu
sub.base$llavepu<-NULL

model<-ksvm(inc~., data=sub.base, type = "C-svc", C=df[bestiter_s4,2],kpar=list(sigma = df[bestiter_s4,3]),prob.model=T)  


#######################
#BASE DE ENTRENAMIENTO#
#######################

preds_train<-predict(model,newdata=sub.base,type="probabilities")
preds_train<-preds_train[,2]
sub.base$preds<-preds_train

library(pROC)
rocobj<-plot.roc(sub.base$inc, sub.base$preds,percent=TRUE, ci=TRUE,print.auc=TRUE) 
rocobj$auc 
#91.43

######################
#BASE DE PRUEBA INDEP#
######################

seg4test<-desarrollo_seg4_test

seg4test$inc<-seg4test$Y2
seg4test$Y2<-NULL

seg4test<-na.omit(seg4test)

seg4_llaves_test<-seg4test$llavepu
seg4test$llavepu<-NULL
seg4test$cosecha<-NULL
seg4test$MODELO<-NULL
seg4test$train<-NULL
seg4test$llavepu<-seg4_llaves_test

#############################
# TRAIN NO UTILIZADA + TEST #
#############################

#68740 observaciones
seg4test_full<-rbind(base_complemento,seg4test)

preds_test<-predict(model,newdata=seg4test_full,type="probabilities")
preds_test<-preds_test[,2]
seg4test_full$preds<-preds_test

library(pROC)
rocobj<-plot.roc(seg4test_full$inc, seg4test_full$preds,percent=TRUE, ci=TRUE,print.auc=TRUE) 
rocobj$auc 
#82.83


###################################
# GUARDANDO BASE DE ENTRENAMIENTO #
###################################

sub.base$llavepu<-seg4_llaves_train
write.csv(sub.base,"seg4_subbase.csv")

write.csv(seg4test_full,"seg4_testfull.csv")

