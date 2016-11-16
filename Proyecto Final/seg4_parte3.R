


supports<-sub.base[model@alphaindex[[1]],]
supports$alpha<-coef(model)[[1]]
#supports_vars<-subset(supports,select=c(r_Apalan,r_Cobert_int,r_liquidez,r_payback,crvtas,vtas_act,deuda_pasivo,cc_activo))
supports_vars<-supports
supports_vars$inc<-NULL
supports_vars$alpha<-NULL
supports_vars$llavepu<-NULL
supports_vars$preds<-NULL

medias<-model@scaling$x.scale$'scaled:center'
sds<-model@scaling$x.scale$'scaled:scale'

supports_sc<-as.matrix(supports_vars)
test<-supports_vars

supports_sc<-scale(supports_sc,center=medias,scale=sds)
test_sc<-scale(test,center=medias,scale=sds)

pred<-predict(model,supports_vars,type="probabilities")
pred<-pred[,2]



#######################################################
# FUNCION PARA CALCULAR LA FUNCION Y LAS PROBAS.
########################################################
probas<-function(test_sc,supports_sc)
{  
  sigma<-model@kernelf@kpar$sigma 
  kerexp<-rep(0,nrow(supports_sc))
  f<-rep(0,nrow(test_sc))
  prob<-rep(0,nrow(test_sc))
  A<-model@prob.model[[1]]$A
  B<-model@prob.model[[1]]$B
  
  for (j in 1:nrow(test_sc)){
    for (i in 1:nrow(supports_sc)){
      aux<-norm(as.matrix(test_sc[j,])-as.matrix(supports_sc[i,]),"F")
      aux<-aux*aux
      kerexp[i]<-exp((-sigma)*(aux))
    }
    f_newobs<-sum(supports$alpha*kerexp)
    f_newobs<-f_newobs-model@b[1]
    prob_newobs<-1/(1+exp(A*f_newobs+B))
    f[j]<-f_newobs
    prob[j]<-prob_newobs
  }
  out<-list(f,prob)
}  
########################################################

out<-probas(test_sc,supports_sc)

f<-out[[1]]
prob<-out[[2]]

supports_probs<-supports
supports_probs$f<-f
supports_probs$prob_hand<-prob
supports_probs$prob_ksvm<-pred

########################################################
#GUARDANDO LOS SUPPORT VECTORS CON SUS RESPECTIVAS ALPHAS

write.csv(supports,"seg4_supports.csv")


