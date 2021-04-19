# Deep Wheels

A demonstrator for predictive maintenance concerning the degradation of car tires.

Deep Wheels is able to predict the Remaining Useful Life (RUL) of car tire profiles based on highly detailed live driving data from the F1 2019 car racing game. 
Live data is temporarily being stored in an Influx database and fed to a Long Short Term Memory (LSTM) recurrent neural network.
The resulting predictions are then visualized using a Grafana web interface.

![Screenshot Deep Wheels](screenshot_deepwheels.jpg?raw=true "Deep Wheels' web interface")
