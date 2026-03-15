Reliance
library(quantmod)    
library(tseries)     
library(forecast)    
library(TTR)         
library(lubridate)   
library(zoo) 

getSymbols("RELIANCE.NS", src = "yahoo", from = "2012-09-07", to = "2025-09-07")
price=RELIANCE.NS$RELIANCE.NS.Close
price=na.locf(price)

# Initial plot
plot(price, main="Reliance Closing Prices", ylab="Price", xlab="Time", col="black")

# Stationarity tests
adf.test(price)
kpss.test(price)

# Transformation and differencing
returns <- diff(log(price))
returns=na.omit(returns)
plot(returns)

# Stationarity on transformed data
adf.test(returns)
kpss.test(returns)

acf(returns,main="Acf of returns")
pacf(returns,main="Pacf of returns")


# Fit candidate ARIMA models, compare AIC/SSE

model1<-arima(returns, order=c(1,0,0))
SSE1<-sum(model1$residuals^2)
AIC(model1)

model2<-arima(returns, order=c(2,0,0))
SSE2<-sum(model2$residuals^2)
AIC(model2)

model3<-arima(returns, order=c(3,0,0))
SSE3<-sum(model3$residuals^2)
AIC(model3)

model4<-arima(returns, order=c(1,0,1))
SSE4<-sum(model4$residuals^2)
AIC(model4)

model5<-arima(returns, order=c(2,0,2))
SSE5<-sum(model5$residuals^2)
AIC(model5)

model6<-arima(returns, order=c(0,0,0))
SSE6<-sum(model6$residuals^2)
AIC(model6)

model=auto.arima(returns)
summary(model)

# Build the results table
results <- data.frame(
  Model = c("ARIMA(1,0,0)", 
            "ARIMA(2,0,0)", 
            "ARIMA(3,0,0)", 
            "ARIMA(1,0,1)", 
            "ARIMA(2,0,2)",
            "ARIMA(0,0,0)"),
  AIC = c(-17041.76, -17040.95, -17039.18, -17039.76, -17054.18, -17043.66),
  SSE = c(SSE1, SSE2, SSE3, SSE4, SSE5,SSE6)
)

# Sort by AIC ascending
results_sorted <- results[order(results$AIC), ]

# View results
print(results_sorted)



# Residual diagnostics
Box.test(residuals(model5), type="Ljung-Box")

#forecasting 

#Model 1
reliance_train <- returns[1:(0.9 * length(returns))]  # Train data set
reliance_test <- returns[(0.9 * length(returns) + 1):length(returns)]  # Test data set
fit <- arima(reliance_train, order = c(2, 0, 2))
arma.preds <- predict(fit, n.ahead = (length(returns) - (0.9 * length(returns))))$pred
reliance.forecast <- forecast(fit, h = 14)
plot(reliance.forecast,xlim=c(2000,2900), main = "ARMA(2,2) forecasts for reliance returns")
accuracy(arma.preds, reliance_test)

#Model 2
reliance_train <- returns[1:(0.9 * length(returns))]  # Train data set
reliance_test <- returns[(0.9 * length(returns) + 1):length(returns)]  # Test data set
fit <- arima(reliance_train, order = c(2, 0, 1))
arma.preds <- predict(fit, n.ahead = (length(returns) - (0.9 * length(returns))))$pred
reliance.forecast <- forecast(fit, h = 14)
plot(reliance.forecast,xlim=c(2500,2900),main = "ARMA(2,1) forecasts for reliance returns")
accuracy(arma.preds, reliance_test)

#Model 3
reliance_train <- returns[1:(0.9 * length(returns))]  # Train data set
reliance_test <- returns[(0.9 * length(returns) + 1):length(returns)]  # Test data set
fit <- arima(reliance_train, order = c(2, 0, 0))
arma.preds <- predict(fit, n.ahead = (length(returns) - (0.9 * length(returns))))$pred
reliance.forecast <- forecast(fit, h = 14)
plot(reliance.forecast,xlim=c(2500,2900), main = "ARMA(2,0) forecasts for reliance returns")
accuracy(arma.preds, reliance_test)

#Model 4
reliance_train <- returns[1:(0.9 * length(returns))]  # Train data set
reliance_test <- returns[(0.9 * length(returns) + 1):length(returns)]  # Test data set
fit <- arima(reliance_train, order = c(1, 0, 2))
arma.preds <- predict(fit, n.ahead = (length(returns) - (0.9 * length(returns))))$pred
reliance.forecast <- forecast(fit, h = 14)
plot(reliance.forecast, xlim=c(2500,2900),main = "ARMA(1,2) forecasts for reliance returns")
accuracy(arma.preds, reliance_test)

#Model 5
reliance_train <- returns[1:(0.9 * length(returns))]  # Train data set
reliance_test <- returns[(0.9 * length(returns) + 1):length(returns)]  # Test data set
fit <- arima(reliance_train, order = c(0, 0, 2))
arma.preds <- predict(fit, n.ahead = (length(returns) - (0.9 * length(returns))))$pred
reliance.forecast <- forecast(fit, h = 14)
plot(reliance.forecast, xlim=c(2500,2900),main = "ARMA(0,2) IN forecasts for reliance returns")
lines(test,lwd = 2) 
accuracy(arma.preds, reliance_test)

# Build comparison table
results <- data.frame(
  Model = c("ARMA(2,2)", "ARMA(2,1)", "ARMA(2,0)", "ARMA(1,2)", "ARMA(0,2)"),
  AIC   = c(-17054.18, -17039.76, -17040.95, -17041.76, -17043.66),  # from earlier
  RMSE  = c(0.0142309, 0.0141737, 0.0141737, 0.0141738, 0.0141738),
  MAE   = c(0.0103608, 0.0102995, 0.0102995, 0.0102995, 0.0102995)
)

# Sort by AIC (best fit first)
results_sorted <- results[order(results$AIC), ]
print(results_sorted, row.names = FALSE)



#out of sample forecasting
fit <- arima(returns, order = c(2, 0, 2))
arma.preds <- predict(fit, n.ahead = (length(returns) - (0.9 * length(returns))))$pred
n.ahead = (length(returns) - (0.9 * length(returns))) 
reliance.forecast <- forecast(fit, h = 14)
plot(reliance.forecast, xlim=c(3000,3300),main = "ARMA out of sample forecasts for reliance returns")
accuracy(arma.preds, reliance_test)
Box.test(residuals(model3), type="Ljung-Box")


