# Title     :
# Objective :
# Created by: sbrood
# Created on: 20/03/19

library(plyr)



smp = read.csv2(file = "/home/sbrood/PROJET_S4_SAT/train_wkt_v4.csv", sep=",") 
x= c()
for(i in c(1:10)){
  t = smp$ClassType == i & smp$MultipolygonWKT != "MULTIPOLYGON EMPTY"
  x[i]=length(grep(TRUE,t))
}

print(x)
pie(x)


