using System;
using System.Collections.Generic;

namespace MLSystemManager
{
	public class VMatrix
	{
		Matrix m_matrix;
		int m_rowCount;
		int m_colStart;
		int m_colCount;
		int[] m_rowData;

		public VMatrix()
		{
			m_matrix = null;
			m_rowCount = 0;
			m_colStart = 0;
			m_colCount = 0;
			m_rowData = null;
		}

		public VMatrix(Matrix matrix, int rowStart, int colStart, int rowCount, int colCount)
		{
			m_matrix = matrix;
			m_rowCount = rowCount;
			m_colStart = colStart;
			m_colCount = colCount;
			m_rowData = new int[rowCount];
			for (var i = 0; i < rowCount; i++)
			{
				m_rowData[i] = i + rowStart;
			}
		}

		public VMatrix(VMatrix matrix, int rowStart, int colStart, int rowCount, int colCount)
		{
			m_matrix = matrix.m_matrix;
			m_rowCount = rowCount;
			m_colStart = colStart + matrix.m_colStart;
			m_colCount = colCount;
			m_rowData = new int[rowCount];
			for (var i = 0; i < rowCount; i++)
			{
				m_rowData[i] = matrix.m_rowData[i + rowStart];
			}
		}

		// Returns the data
		public List<double[]> Data { get { return m_matrix == null ? null : m_matrix.Data; } }

		// Returns the rows array in the matrix
		public int[] RowData() { return m_rowData; }

		// Returns the number of rows in the matrix
		public int Rows() { return m_rowCount; }

		// Returns the number of columns (or attributes) in the matrix
		public int Cols() { return m_colCount; }

		// Returns the specified row
		public double[] Row(int r) { return m_matrix.Row(m_rowData[r]); }

		// Returns the element at the specified row and column
		public double Get(int r, int c) { return m_matrix.Get(m_rowData[r], c + m_colStart); }

		// Sets the value at the specified row and column
		public void Set(int r, int c, double v) { m_matrix.Set(m_rowData[r], c + m_colStart, v); }

		// Returns the name of the specified attribute
		public String AttrName(int col) { return m_matrix.AttrName(col + m_colStart); }

		// Set the name of the specified attribute
		public void SetAttrName(int col, String name) { m_matrix.SetAttrName(col + m_colStart, name); }

		// Returns the name of the specified value
		public String AttrValue(int attr, int val) { return m_matrix.AttrValue(attr + m_colStart, val); }

		// Returns the number of values associated with the specified attribute (or column)
		// 0=continuous, 2=binary, 3=trinary, etc.
		public int ValueCount(int col) { return m_matrix.ValueCount(col + m_colStart); }

		// Shuffles the row order
		public void Shuffle(Random rand)
		{
			for (var n = Rows(); n > 0; n--)
			{
				var i = rand.Next(n);
				var tmp = m_rowData[n - 1];
				m_rowData[n - 1] = m_rowData[i];
				m_rowData[i] = tmp;
			}
		}

		// Shuffles the row order with a buddy matrix 
		public void Shuffle(Random rand, VMatrix buddy)
		{
			for (var n = Rows(); n > 0; n--)
			{
				var i = rand.Next(n);
				var tmp = m_rowData[n - 1];
				m_rowData[n - 1] = m_rowData[i];
				m_rowData[i] = tmp;

				if (buddy != null)
				{
					var tmp1 = buddy.RowData()[n - 1];
					buddy.RowData()[n - 1] = buddy.RowData()[i];
					buddy.RowData()[i] = tmp1;
				}
			}
		}

		// Returns the mean of the specified column
		public double ColumnMean(int col)
		{
			return m_matrix.ColumnMean(col + m_colStart);
		}

		// Returns the min value in the specified column
		public double ColumnMin(int col)
		{
			return m_matrix.ColumnMin(col + m_colStart);
		}

		// Returns the max value in the specified column
		public double ColumnMax(int col)
		{
			return m_matrix.ColumnMax(col + m_colStart);
		}

		// Returns the original min value in the specified column
		public double ColumnMinOrig(int col)
		{
			return m_matrix.ColumnMinOrig(col + m_colStart);
		}

		// Returns the max value in the specified column
		public double ColumnMaxOrig(int col)
		{
			return m_matrix.ColumnMaxOrig(col + m_colStart);
		}

		// Returns the most common value in the specified column
		public double MostCommonValue(int col)
		{
			return m_matrix.MostCommonValue(col + m_colStart);
		}
	}
}
