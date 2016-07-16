from Reader import Reader
from DigitOCR import DigitOCR
from Solver import solve_grid

ocm = DigitOCR()

ocm.create_training_data()
ocm.train()


reader = Reader()



reader.load_image("images/sudoku6.jpg")
reader.clean_image()
reader.show_image()
reader.rectify_perspective()
reader.show_rectified()
reader.cut_image_from_clean()


reader.margin_cases()


mat = reader.convert_to_matrix(ocm)
print mat

print "Searching for a solution..."
print solve_grid(mat)

