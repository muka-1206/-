from model import BoidFlockers
from ModelRunner import ModelRunner

param_path = "./parameter/nominal.toml"
runner = ModelRunner(BoidFlockers, param_path)
runner.save("./movie/movie.gif", writer="pillow")
runner.make_stackplot("./figure/stackplot.png")
