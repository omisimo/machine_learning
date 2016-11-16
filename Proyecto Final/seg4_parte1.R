

sampling<-function(base,T){
  
  library(mlr)
  library(kernlab)
  library(ROCR)
  #library(pROC)
  ################################
  #Argumentos para la funcion MLR
  learner = makeLearner("classif.ksvm",predict.type="prob")
  # SE DEFINE EL CONJUNTO DE VALORES PARA FORMAR EL GRID
  param= makeParamSet( 
    makeDiscreteParam("C", values = c(1)), 
    makeDiscreteParam("sigma", values = c(.1))
  )
  ctrl = makeTuneControlGrid()
  inner = makeResampleDesc("Holdout")       #Holdout validation  
  outer = makeResampleDesc("CV", iters = 3) #K-fold CrossValidation
  
  ################################
  malos<-subset(base,inc==1)   #Subset con clientes con incumplimiento igual a 1
  num_malos<-nrow(malos)
  buenos<-subset(base,inc==0)  #Subset con clientes con incumplimiento igual a 0
  
  #En estos vectores vamos a guardar los mejores AUC por VC, los mejores parametros C
  #y sigma en cada iteracion y el AUC con muestra de prueba
  aucs<-rep(0,T)
  Cs<-rep(0,T)
  sigmas<-rep(0,T)
  aucs_train<-rep(0,T)
  ci_lower<-rep(0,T)
  ci_upper<-rep(0,T)
  nsv<-rep(0,T) #Numero de SV
  tprs<-rep(0,T)
  accs<-rep(0,T)
  
  #Loop:
  # Hacemos T iteraciones. COnservamos los malos
  for (i in 1:T){
    library(mlr)
    set.seed(i)
    
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
    task<-makeClassifTask(data=sub.base, target="inc")
    lrnr_task = makeTuneWrapper(learner, resampling = outer, par.set = param, 
                                control = ctrl, measure=list(auc,tpr,fpr,acc))
    set.seed(i)
    mod = train(lrnr_task, task)
    
    aucs[i]<-mod$learner.model$opt.result$y[1]
    tprs[i]<-mod$learner.model$opt.result$y[2]
    accs[i]<-mod$learner.model$opt.result$y[4]
    Cs[i]<-mod$learner.model$opt.result$x[[1]]
    sigmas[i]<-mod$learner.model$opt.result$x[[2]]
    
    
    
    model<-ksvm(inc~., data=sub.base, type = "C-svc", C=mod$learner.model$opt.result$x[[1]],
                kpar=list(sigma = mod$learner.model$opt.result$x[[2]]),prob.model=T)  
    nsv[i]<-model@nSV
    
    set.seed(i)
    
    preds_train<-predict(model,newdata=base_complemento,type="probabilities")
    preds_train<-preds_train[,2]
    base_complemento$preds<-preds_train
    
    library(pROC)
    rocobj<-plot.roc(base_complemento$inc, base_complemento$preds,percent=TRUE, 
                     ci=TRUE,print.auc=TRUE) 
    aucs_train[i]<-rocobj$auc
    
    detach("package:pROC",unload=T)
    detach("package:mlr",unload=T)
    
  }
  #Guardamos los vectores de los resultados en una lista y retornamos ese objeto
  list_result<-list(aucs,Cs,sigmas,aucs_train,tprs,accs,nsv)                                           
  return(list_result)
}

################################################################################
# SEGMENTO 4
################################################################################

#43280 clientes Bancomer
clientes_bm<-desarrollo_seg4_train[1:43280,]
clientes_nobm<-desarrollo_seg4_train[43281:87228,]

#nrow(clientes_nobm_malos)/nrow(clientes_nobm)
clientes_nobm_malos<-subset(clientes_nobm,Y2==1)

nrow(clientes_nobm_malos)/nrow(clientes_nobm)

#51260 observaciones
seg4_new<-rbind(clientes_bm,clientes_nobm_malos)

seg4_llaves_new<-seg4_new$llavepu
seg4_new$llavepu<-NULL
seg4_new$train<-NULL
seg4_new$MODELO<-NULL
seg4_new$cosecha<-NULL

seg4_new$inc<-seg4_new$Y2
seg4_new$Y2<-NULL

result_new<-sampling(seg4_new,125)

df<-as.data.frame(cbind(result_new[[1]],result_new[[2]],result_new[[3]],
                        result_new[[4]],result_new[[5]],result_new[[6]],
                        result_new[[7]]))

names(df)<-c("AUC_CV","C","sigma","AUC_Test","TPR","ACC","Nsv")

write.csv(df,"result_seg4.csv")