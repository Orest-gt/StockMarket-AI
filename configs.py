num_epochs = 50
ticker = "NVDA"
best_val_loss = float("inf")
patience = 15
wait = 0

main_training_date_start = "2020-01-01"
main_training_date_end = "2025-05-29"
add_training_date_start = "2000-01-01"
add_training_data_end = "2019-12-31"
predict_data_period = "1d"
best_model_path = r"C:\Users\Orestis\PycharmProjects\StockMarket-AI\tests\best_model_" + ticker + "-"
best_model_path_end = ".pth"
print("  " + best_model_path + "data" + best_model_path_end)