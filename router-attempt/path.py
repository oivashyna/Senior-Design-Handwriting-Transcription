import os


project_root = os.path.abspath((os.path.dirname(__file__)))
print(project_root)

project_root2 = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../..")
            )
print(project_root2)