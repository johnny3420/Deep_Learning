# Load Libraries
library(tidyverse)
library(lubridate)
library(keras)
use_condaenv('r-reticulate')
library(tensorflow)

# Start GPU
physical_devices <- tf$config$list_physical_devices('GPU')
tf$config$experimental$set_memory_growth(device = physical_devices[[1]], enable = T)

# Data
base_dir <- "D:/R_Stuff/Deep_Learning/cats_and_dogs_small"
train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")

# Flags

FLAGS <- flags(
  flag_integer("kernel", 3, "kernel size"),
  flag_integer("filters", 32, "number of filters in first layer"),
  flag_numeric("dropout", 0.5, "dropout rate")
)

# Book Model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = FLAGS$filters, kernel_size = c(FLAGS$kernel, FLAGS$kernel), activation = "relu",
                input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = FLAGS$filters*2, kernel_size = c(FLAGS$kernel, FLAGS$kernel), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = FLAGS$filters*2*2, kernel_size = c(FLAGS$kernel, FLAGS$kernel), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = FLAGS$filters*2*2, kernel_size = c(FLAGS$kernel, FLAGS$kernel), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dropout(rate = FLAGS$dropout) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("acc")
)

# Generators
datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE
)
test_datagen <- image_data_generator(rescale = 1/255)
train_generator <- flow_images_from_directory(train_dir,
                                              datagen,
                                              target_size = c(150, 150),
                                              batch_size = 16,
                                              class_mode = "binary"
                                              )
validation_generator <- flow_images_from_directory(
  validation_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 16,
  class_mode = "binary"
)

test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 16,
  class_mode = "binary"
)

# Run model
history <- model %>% fit(
  train_generator,
  steps_per_epoch = 100,
  epochs = 50,
  validation_data = validation_generator,
  validation_steps = 50
)

plot(history)

# Scores
score <- model %>% evaluate(test_generator, steps = 50)

#cat('Test loss:', score$loss, '\n')
#cat('Test accuracy:', score$acc, '\n')