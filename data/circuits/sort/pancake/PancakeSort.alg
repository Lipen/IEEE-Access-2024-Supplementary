// Pancake Sort
// Data array of length `k`
// Each sorted number is `n`-bit

define k 9;
define n 4;

__in bit data[k][n];

__out bit sorted_data[k][n];

void Flip(int p) {
	for (int i = 0; i < p; i = i + 1) {
		p = p - 1;
		bit tmp[n] = data[i];
		data[i] = data[p];
		data[p] = tmp;
	}
}

void main() {
  	for (int i = k - 1; i >= 0; i = i - 1) {
		bit max[4] = 0;
		for (int j = 1; j <= i; j = j + 1) {
			// if (data[j] > data[max]) max = j;
			// max can be from 0 to i (inclusive)
			for(int d = 0; d <= i; d = d + 1) {
				if ((max == d) & (data[d] < data[j])) {
					max = j;
				}
			}
		}

		for (int t = 1; t < i; t = t + 1) {
			if (max == t) {
				Flip(t + 1);
			}
		}

		if (max != i) {
			Flip(i + 1);
		}
	}

	sorted_data = data;
}
