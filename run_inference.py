import mamt

if __name__ == "__main__":
    mamt.do_inference(
        dir = "./inference",
        output_dir = "./inference",
        checkpoint_path = "./model/model_3.pt",
    )