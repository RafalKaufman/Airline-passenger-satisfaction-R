#Wczytanie danych z plików CSV
aircustomers_train_default <- read.csv("train.csv", stringsAsFactors = TRUE)
aircustomers_test_default <- read.csv("test.csv", stringsAsFactors = TRUE)

#Połączenie obserwacji w jedną ramkę danych
aircustomers <- rbind(aircustomers_train_default, aircustomers_test_default)

#Sprawdzenie poprawności załadowania danych
str(aircustomers)
summary(aircustomers)

#Usunięcie nadmiarowej cechy id oraz mocno skorelowanej cechy Arrival.Delay.in.Minutes
cor.test(aircustomers$Departure.Delay.in.Minutes, aircustomers$Arrival.Delay.in.Minutes)
cor.test(aircustomers$On.board.service, aircustomers$Inflight.service)
aircustomers = subset(aircustomers, select = -c(id,Arrival.Delay.in.Minutes))
str(aircustomers)

#Zamiana zer w zmiennych jakościowych na NA
aircustomers$Inflight.wifi.service<-ifelse(aircustomers$Inflight.wifi.service==0, NA, aircustomers$Inflight.wifi.service)
aircustomers$Departure.Arrival.time.convenient<-ifelse(aircustomers$Departure.Arrival.time.convenient==0, NA, aircustomers$Departure.Arrival.time.convenient)
aircustomers$Ease.of.Online.booking<-ifelse(aircustomers$Ease.of.Online.booking==0, NA, aircustomers$Ease.of.Online.booking)
aircustomers$Gate.location<-ifelse(aircustomers$Gate.location==0, NA, aircustomers$Gate.location)
aircustomers$Food.and.drink<-ifelse(aircustomers$Food.and.drink==0, NA, aircustomers$Food.and.drink)
aircustomers$Online.boarding<-ifelse(aircustomers$Online.boarding==0, NA, aircustomers$Online.boarding)
aircustomers$Seat.comfort<-ifelse(aircustomers$Seat.comfort==0, NA, aircustomers$Seat.comfort)
aircustomers$Inflight.entertainment<-ifelse(aircustomers$Inflight.entertainment==0, NA, aircustomers$Inflight.entertainment)
aircustomers$On.board.service<-ifelse(aircustomers$On.board.service==0, NA, aircustomers$On.board.service)
aircustomers$Leg.room.service<-ifelse(aircustomers$Leg.room.service==0, NA, aircustomers$Leg.room.service)
aircustomers$Baggage.handling<-ifelse(aircustomers$Baggage.handling==0, NA, aircustomers$Baggage.handling)
aircustomers$Checkin.service<-ifelse(aircustomers$Checkin.service==0, NA, aircustomers$Checkin.service)
aircustomers$Inflight.service<-ifelse(aircustomers$Inflight.service==0, NA, aircustomers$Inflight.service)
aircustomers$Cleanliness<-ifelse(aircustomers$Cleanliness==0, NA, aircustomers$Cleanliness)

#Sprawdzenie i usunięcie missing values
sum(is.na(aircustomers$Inflight.wifi.service))/nrow(aircustomers)
sum(is.na(aircustomers$Departure.Arrival.time.convenient))/nrow(aircustomers)
sum(is.na(aircustomers$Ease.of.Online.booking))/nrow(aircustomers)
sum(is.na(aircustomers$Gate.location))/nrow(aircustomers)
sum(is.na(aircustomers$Food.and.drink))/nrow(aircustomers)
sum(is.na(aircustomers$Online.boarding))/nrow(aircustomers)
sum(is.na(aircustomers$Seat.comfort))/nrow(aircustomers)
sum(is.na(aircustomers$Inflight.entertainment))/nrow(aircustomers)
sum(is.na(aircustomers$On.board.service))/nrow(aircustomers)
sum(is.na(aircustomers$Leg.room.service))/nrow(aircustomers)
sum(is.na(aircustomers$Baggage.handling))/nrow(aircustomers)
sum(is.na(aircustomers$Checkin.service))/nrow(aircustomers)
sum(is.na(aircustomers$Inflight.service))/nrow(aircustomers)
sum(is.na(aircustomers$Cleanliness))/nrow(aircustomers)
install.packages("tidyr")
library("tidyr")
aircustomers <- drop_na(aircustomers)
summary(aircustomers)
nrow(aircustomers)

#Kosmetyczna zmiana nazwy X na id
names(aircustomers)[names(aircustomers) == 'X'] <- 'id'
aircustomers_model <- aircustomers
str(aircustomers_model)

#Encoding binarny zmiennych jakościowych o charakterze binarnym
aircustomers_model$Gender <- ifelse(aircustomers_model$Gender == "Female",1,0)
aircustomers_model$Customer.Type <- ifelse(aircustomers_model$Customer.Type == "disloyal Customer",1,0)
aircustomers_model$Type.of.Travel <- ifelse(aircustomers_model$Type.of.Travel == "Business travel",1,0)
aircustomers_model$satisfaction <- ifelse(aircustomers_model$satisfaction == "satisfied",1,0)
str(aircustomers_model)

#One-hot encoding zmiennych nominalnych
install.packages("recipes", dependencies = TRUE)
install.packages("caret")
library("caret")
dummy <- dummyVars(" ~ .", data = aircustomers_model, fullRank = T)
aircustomers_transformed <- data.frame(predict(dummy, newdata = aircustomers_model))
str(aircustomers_transformed)

#Normalizacja min-max
variables <- aircustomers_transformed[2:24]
normalize <- function(x){
  return ((x-min(x))/(max(x)-min(x)))
}
variables_normalize <- as.data.frame(lapply(variables, normalize))
summary(variables$Departure.Delay.in.Minutes)
summary(variables_normalize$Departure.Delay.in.Minutes)
summary(variables_normalize)

#Budowanie modelu klasteryzacji k-średnich
set.seed(123)
k <- kmeans(variables_normalize, centers = 2, nstart = 25)
str(k)

sum_squared_errors <- function(k) {
  kmeans(variables_normalize, k, nstart = 10 )$tot.withinss
}
k.values <- 1:15
library("purrr")
sum_squared_errors_values <- map(k.values, sum_squared_errors)

#Wybór najbardziej optymalnego k metodą łokcia
plot(k.values, sum_squared_errors_values, type="b", xlab="Number of clusters K", ylab="Total within-clusters sum of squares")

#Finalny model
aircustomers_clusters <- kmeans(variables_normalize, centers = 3, nstart = 25)
str(aircustomers_clusters)
aircustomers_clusters$size

#Dołączenie kolumny z numerem klastra do wyjściowej ramki danych
aircustomers$cluster <- aircustomers_clusters$cluster
str(aircustomers)

#Analiza centroidów i cech dominujących w klastrach
aircustomers_clusters$centers
aggregate(data = aircustomers, Seat.comfort ~ cluster, mean)
aggregate(data = aircustomers, Inflight.entertainment ~ cluster, mean)

#Wymieszanie danych oraz podział na zbiór uczący i testowy pod model klasyfikacyjny
train_sample <- sample(119567, 83697)
str(train_sample)
str(aircustomers)
aircustomers_tree <- aircustomers[-c(1,24)]
str(aircustomers_tree)
aircustomers_train <- aircustomers_tree[train_sample, ]
aircustomers_test <- aircustomers_tree[-train_sample, ]

#Sprawdzenie podziału zmiennej klasyfikującej
prop.table(table(aircustomers_train$satisfaction))
prop.table(table(aircustomers_test$satisfaction))

#Budowa modelu klasyfikacyjnego drzewa losowego
library("C50")
library("gmodels")
aircustomers_model_tree <- C5.0(aircustomers_train[-22], aircustomers_train$satisfaction)
aircustomers_model_tree
summary(aircustomers_model_tree)
aircustomers_tree_pred <- predict(aircustomers_model_tree, aircustomers_test)

#Ocena modelu używając tablicy krzyżowej
CrossTable(aircustomers_test$satisfaction, aircustomers_tree_pred, prop.chiq=FALSE, dnn=c('actual satisfaction', 'predicted satisfaction'))