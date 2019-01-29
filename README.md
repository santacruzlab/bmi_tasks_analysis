BMI3D tasks and analysis
========================
This repository is meant to contain specific tasks for Carmena lab tasks and code for conducting some parts of analysis used in previous studies. It should remain internal, i.e., private and not public.

Using with bmi3d
================
Symbolically link the tasks and analysis folders into the bmi3d folder:

```
  cd $HOME/code
  git clone https://github.com/carmenalab/brain-python-interface.git bmi3d
  git clone https://github.com/carmenalab/bmi3d_tasks_analysis.git
  cd bmi3d
  ln -s ../bmi3d_tasks_analysis/analysis/ analysis
  ln -s ../bmi3d_tasks_analysis/tasks/ tasks
```
