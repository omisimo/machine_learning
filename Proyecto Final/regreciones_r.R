

########################################################################
# REGRESIONES LOGISTICAS: SEGMENTO 1 Y 4
# Se ajustan regresiones logisticas para comparar el desempe~no con los
# modelos obtenidos mediante SVM
########################################################################

seg4_train<-desarrollo_seg4_train
seg4_train$llavepu<-NULL
seg4_train$cosecha<-NULL
seg4_train$MODELO<-NULL
seg4_train$train<-NULL
seg4_reglog <- glm(Y2 ~ .,family = binomial, data = seg4_train)
# INFORMACION SOBRE EL MODELO (BETAS, P-VALUES, ETC.)
summary(seg4_reglog)
#######################
# BASE DE ENTRENAMIENTO
#######################
pd_reglog<-predict(seg4_reglog,newdata=seg4_train,type="response")
seg4_train$pd_reglog<-pd_reglog

roccurve<-plot.roc(seg4_train$Y2, seg4_train$pd_reglog,
                   percent=TRUE, ci=TRUE,print.auc=TRUE)
roccurve$auc

#######################
# BASE DE PRUEBA
#######################
seg4_test<-desarrollo_seg4_test
seg4_test$llavepu<-NULL
seg4_test$cosecha<-NULL
seg4_test$MODELO<-NULL
seg4_test$train<-NULL
pd_reglog<-predict(seg4_reglog,newdata=seg4_test,type="response")
seg4_test$pd_reglog<-pd_reglog
roccurve<-plot.roc(seg4_test$Y2, seg4_test$pd_reglog,
                   percent=TRUE, ci=TRUE,print.auc=TRUE)
roccurve$auc
#78.61


