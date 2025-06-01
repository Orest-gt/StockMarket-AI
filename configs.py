num_epochs = 200
ticker = "NVDA"
best_val_loss = float("inf")
patience = 20

live_training = False
main_training_date_start = "2023-01-01"
main_training_date_end = "2025-05-30"
add_training_date_start = "2024-12-31"
add_training_data_end = "2025-05-30"
predict_data_period = "1d"
best_model_path = r"tests" + r"/best_model_" + ticker + "-"
best_model_path_end = ".pth"
print("  " + best_model_path + "data" + best_model_path_end)
epsilon = 1-e4

'''
2025-05-31-12-47
2025-05-31-12-48
2025-05-31-12-49
2025-05-31-12-50
2025-05-31-12-51
'''
