from rectpack import newPacker


class RECTANGLE_PACKER:

	def __init__(self, rectangles, rectangle_names, bins):
		self.packer = newPacker()
		# Add the rectangles to packing queue
		for i, r in enumerate(rectangles):
			self.packer.add_rect(*r, rid=rectangle_names[i])
		# Add the bins where the rectangles will be placed
		for b in bins:
			self.packer.add_bin(*b)

	def solve(self, show_result=False):
		"""

		:return: dictionary with key, value pairs of the form rid: (x, y, w, h, b) where rid is a rectangle id,
				x = rectangle bottom-left x coordinate
				y = rectangle bottom-left y coordinate
				w = rectangle width
				h = rectangle height
				b = bin index
		"""
		# Start packing
		self.packer.pack()
		# Obtain number of bins used for packing
		nbins = len(self.packer)
		# if no bin is used
		if nbins == 0:
			# no solution is found
			return None
		# otherwise
		else:
			# Number of rectangles packed into first bin
			num_rect_packed = len(self.packer[0])
			# Number of rectangles to pack
			num_rect_to_pack = len(self.packer._avail_rect)
			# if all the rectangles could not be packed in the first bin
			if num_rect_packed < num_rect_to_pack:
				# no solution is found
				return None
			# a solution has been found
			else:
				# Initialize dictionary for the solution
				arrangement_dict = {}
				# Full rectangle list
				all_rects = self.packer.rect_list()
				for rect in all_rects:
					b, x, y, w, h, rid = rect
					arrangement_dict.update({rid: (x, y, w, h, b)})
					# b - Bin index
					# x - Rectangle bottom-left corner x coordinate
					# y - Rectangle bottom-left corner y coordinate
					# w - Rectangle width
					# h - Rectangle height
					# rid - User assigned rectangle id or None
					if show_result:
						print('Place bottom left corner of rectangle #' + str(rid) + ' of size W' + str(w) + 'xH' + str(h) + ' at (' + str(x) + ',' + str(y) + ') in bin ' + str(b))
				return arrangement_dict




