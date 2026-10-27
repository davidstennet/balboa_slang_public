__Custom Scene - HW 2_8__

Scene name: Phase 1 Clone Trooper Helmet (Stylized)
Scene location: scenes/hw2/clone_helmet_spin.json
Rendering location: outputs/hw_2_8.png
Mesh files: scenes/hw2/clone_trooper/
Mesh generation script: make_clone_helmet.py

__How to Run__

Generate the meshes and scene:
uv run python make_clone_helmet.py

View the animation:
uv run python main.py 2_7 scenes/hw2/clone_helmet_spin.json

Generate the submitted rendering:
uv run python main.py -output outputs/hw_2_8.png 2_7 -time 0.0 scenes/hw2/clone_helmet_spin.json

External Assets
No external 3D models or textures were used. Helmet meshes were generated procedurally. Design was inspired by Star Wars Phase 1 clone trooper helmet references.

Bonus
No bonus features were implemented.