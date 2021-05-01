from python.test.test_detectron import run
import os
if __name__ == "__main__":
    envs = list(os.environ.items())
    envs = sorted(envs, key=lambda x: x[0])

    for idx, (k, v) in enumerate(envs):
        print(f"{idx} \t {k} \t {v}")
        
    print("stocazzo")
    run()


