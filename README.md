- train.py: combinition of drqa train and squad train
- train_drqa.py: original(? maybe some minor modification) train code from drqa
- args.py: args of drqa (need modification, add more sqaud args)
- setup.py: original squad setup
- prepro.py: drqa setup (based on setup.py)
- setup_drqa: put drqa code into the structure of squad (working on right now)



Previous steps of data preprocess: setup.py -> prepro.py -> train.py

Wanted: setup_drqa.py -> train_drqa.py -> test_drqa.py

Want new files with postfix "_drqa": fit drqa codes into the structure of original squad
