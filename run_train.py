import mamt

if __name__ == "__main__":    
    result = mamt.do_train(
        mel_dir  = "./dataset/input",
        tgt_dir = "./dataset/target",
        epochs = 20,
        batch_size = 8,
        lr = 1e-4,
        train_ratio = 0.85,
        val_ratio = 0.1,
        test_ratio = 0.05,
        pretrained_path = None,
    )
    print(result)