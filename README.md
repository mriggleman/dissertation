Readme 
#
# SETUP: Ensure the file structure matches this:
#
#   /MyDrive/MAIN_Irish/
#       ├── main_irish_scorer-4.py      ← the scorer
#       └── transcripts/
#               ├── P01_dog.txt
#               ├── P02_cat.txt
#               └── ...
#
# FILE NAMING: each transcript must include the story name in its filename:
#   dog / cat / birds / goats
#   e.g.  21MNM1FC03_dog.txt   or   child01_cat.txt
#   This is how the scorer knows which sets to use when scoring. 
#
# TRANSCRIPT FORMAT: CHAT-style .txt files with *SP01: / *SP02: speaker lines.
#   The scorer automatically extracts only the child's turns (SP02).
#   Comprehension questions and answers within the same file are also scored.
