// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;

namespace MLSystemManager
{
	public class Matrix
	{
		// Data
		List<double[]> m_data;

		// Meta-data
		List<String> m_attr_name;
		List<double> m_minOrig;
		List<double> m_maxOrig;
		List<Dictionary<String, int>> m_str_to_enum;
		List<Dictionary<int, String>> m_enum_to_str;

		public static double MISSING = Double.MaxValue; // representation of missing values in the dataset

		// Creates a 0x0 matrix. You should call loadARFF or setSize next.
		public Matrix() { }

		// Copies the specified portion of that matrix into this matrix
		public Matrix(Matrix that, int rowStart, int colStart, int rowCount, int colCount)
		{
			m_data = new List<double[]>();
			for (int j = 0; j < rowCount; j++)
			{
				double[] rowSrc = that.Row(rowStart + j);
				double[] rowDest = new double[colCount];
				for (int i = 0; i < colCount; i++)
					rowDest[i] = rowSrc[colStart + i];
				m_data.Add(rowDest);
			}
			m_attr_name = new List<String>();
			m_minOrig = new List<double>();
			m_maxOrig = new List<double>();
			m_str_to_enum = new List<Dictionary<String, int>>();
			m_enum_to_str = new List<Dictionary<int, String>>();
			for (int i = 0; i < colCount; i++)
			{
				m_attr_name.Add(that.AttrName(colStart + i));
				m_str_to_enum.Add(that.m_str_to_enum[colStart + i]);
				m_enum_to_str.Add(that.m_enum_to_str[colStart + i]);
				m_minOrig.Add(ColumnMin(i));
				m_maxOrig.Add(ColumnMax(i));
			}
		}

		// Adds a copy of the specified portion of that matrix to this matrix
		public void Add(Matrix that, int rowStart, int colStart, int rowCount)
		{
			if (colStart + Cols() > that.Cols())
			{
				throw new Exception("out of range");
			}
			for (int i = 0; i < Cols(); i++)
			{
				if (that.ValueCount(colStart + i) != ValueCount(i))
				{
					throw new Exception("incompatible relations");
				}
			}
			for (int j = 0; j < rowCount; j++)
			{
				double[] rowSrc = that.Row(rowStart + j);
				double[] rowDest = new double[Cols()];
				for (int i = 0; i < Cols(); i++)
				{
					rowDest[i] = rowSrc[colStart + i];
				}
				m_data.Add(rowDest);
			}
		}

		// Resizes this matrix (and sets all attributes to be continuous)
		public void SetSize(int rows, int cols)
		{
			m_data = new List<double[]>();
			for (int j = 0; j < rows; j++)
			{
				double[] row = new double[cols];
				m_data.Add(row);
			}
			m_attr_name = new List<String>();
			m_str_to_enum = new List<Dictionary<String, int>>();
			m_enum_to_str = new List<Dictionary<int, String>>();
			for (int i = 0; i < cols; i++)
			{
				m_attr_name.Add("");
				m_str_to_enum.Add(new Dictionary<String, int>());
				m_enum_to_str.Add(new Dictionary<int, String>());
			}
		}

		// Loads from an ARFF file
		public void LoadArff(String filename)
		{
			m_data = new List<double[]>();
			m_attr_name = new List<String>();
			m_minOrig = new List<double>();
			m_maxOrig = new List<double>();
			m_str_to_enum = new List<Dictionary<String, int>>();
			m_enum_to_str = new List<Dictionary<int, String>>();
			bool READDATA = false;
			StreamReader file = new StreamReader(filename);
			String line;
			while ((line = file.ReadLine()) != null)
			{
				if (line.Length > 0 && line[0] != '%')
				{
					if (!READDATA)
					{
						Scanner t = new Scanner(line);
						String firstToken = t.Next().ToUpper();

						if (firstToken.Equals("@RELATION"))
						{
							String datasetName = t.Next();
						}

						else if (firstToken.Equals("@ATTRIBUTE"))
						{
							Dictionary<String, int> ste = new Dictionary<String, int>();
							m_str_to_enum.Add(ste);
							Dictionary<int, String> ets = new Dictionary<int, String>();
							m_enum_to_str.Add(ets);

							Scanner u = new Scanner(line);
							if (line.IndexOf("'") >= 0)
							{
								u.UseDelimiter('\'');
							}
							u.Next();
							String attributeName = u.Next();
							if (line.IndexOf("'") >= 0)
							{
								attributeName = "'" + attributeName + "'";
							}
							m_attr_name.Add(attributeName);

							int vals = 0;
							String type = u.Next().Trim().ToUpper();
							if (type.Equals("REAL") || type.Equals("CONTINUOUS") || type.Equals("INTEGER"))
							{
							}
							else
							{
								try
								{
									String values = line.Substring(line.IndexOf("{") + 1, line.IndexOf("}") - line.IndexOf("{") - 1);
									Scanner v = new Scanner(values);
									v.UseDelimiter(',');
									while (v.HasNext())
									{
										String value = v.Next().Trim();
										if (value.Length > 0)
										{
											ste.Add(value, vals);
											ets.Add(vals, value);
											vals++;
										}
									}
								}
								catch (Exception e)
								{
									throw new Exception("Error parsing line: " + line + "\n" + e.ToString());
								}
							}
						}
						else if (firstToken.Equals("@DATA"))
						{
							READDATA = true;
						}
					}
					else
					{
						double[] newrow = new double[Cols()];
						int curPos = 0;

						try
						{
							Scanner t = new Scanner(line);
							t.UseDelimiter(',');
							while (t.HasNext())
							{
								String textValue = t.Next().Trim();

								if (textValue.Length > 0)
								{
									double doubleValue;
									int vals = m_enum_to_str[curPos].Count;

									//Missing instances appear in the dataset as a double defined as MISSING
									if (textValue.Equals("?"))
									{
										doubleValue = MISSING;
									}
									// Continuous values appear in the instance vector as they are
									else if (vals == 0)
									{
										doubleValue = Double.Parse(textValue);
									}
									// Discrete values appear as an index to the "name" 
									// of that value in the "attributeValue" structure
									else
									{
										doubleValue = m_str_to_enum[curPos][textValue];
										if (doubleValue == -1)
										{
											throw new Exception("Error parsing the value '" + textValue + "' on line: " + line);
										}
									}

									newrow[curPos] = doubleValue;
									curPos++;
								}
							}
						}
						catch (Exception e)
						{
							throw new Exception("Error parsing line: " + line + "\n" + e.ToString());
						}
						m_data.Add(newrow);
					}
				}
			}

			file.Close();

			// save the min and max
			for (var col = 0; col < Cols(); col++)
			{
				m_minOrig.Add(ColumnMin(col));
				m_maxOrig.Add(ColumnMax(col));
			}
		}

		// Returns the data
		public List<double[]> Data { get { return m_data; } }

		// Returns the number of rows in the matrix
		public int Rows() { return m_data.Count; }

		// Returns the number of columns (or attributes) in the matrix
		public int Cols() { return m_attr_name.Count; }

		// Returns the specified row
		public double[] Row(int r) { return m_data[r]; }

		// Returns the element at the specified row and column
		public double Get(int r, int c) { return m_data[r][c]; }

		// Sets the value at the specified row and column
		public void Set(int r, int c, double v) { Row(r)[c] = v; }

		// Appends a row
		public void Append(double[] row) { m_data.Add(row); }

		// Inserts a row at the specified location
		public void Insert(int index, double[] row) { m_data.Insert(index, row); }

		// Deletes the specified row
		public void Delete(int index) { m_data.RemoveAt(index); }

		// Returns the name of the specified attribute
		public String AttrName(int col) { return m_attr_name[col]; }

		// Set the name of the specified attribute
		public void SetAttrName(int col, String name) { m_attr_name[col] = name; }

		// Returns the name of the specified value
		public String AttrValue(int attr, int val) { return m_enum_to_str[attr][val]; }

		// Returns the number of values associated with the specified attribute (or column)
		// 0=continuous, 2=binary, 3=trinary, etc.
		public int ValueCount(int col) { return m_enum_to_str[col].Count; }

		// Shuffles the row order
		public void Shuffle(Random rand)
		{
			for (int n = Rows(); n > 0; n--)
			{
				int i = rand.Next(n);
				double[] tmp = Row(n - 1);
				m_data[n - 1] = Row(i);
				m_data[i] = tmp;
			}
		}

		// Shuffles the row order with a buddy matrix 
		public void Shuffle(Random rand, Matrix buddy)
		{
			for (int n = Rows(); n > 0; n--)
			{
				int i = rand.Next(n);
				double[] tmp = Row(n - 1);
				m_data[n - 1] = Row(i);
				m_data[i] = tmp;

				if (buddy != null)
				{
					double[] tmp1 = buddy.Row(n - 1);
					buddy.m_data[n - 1] = buddy.Row(i);
					buddy.m_data[i] = tmp1;
				}
			}
		}

		// Returns the mean of the specified column
		public double ColumnMean(int col)
		{
			double sum = 0;
			int count = 0;
			for (int i = 0; i < Rows(); i++)
			{
				double v = Get(i, col);
				if (v != MISSING)
				{
					sum += v;
					count++;
				}
			}
			return sum / count;
		}

		// Returns the min value in the specified column
		public double ColumnMin(int col)
		{
			double m = MISSING;
			for (int i = 0; i < Rows(); i++)
			{
				double v = Get(i, col);
				if (v != MISSING)
				{
					if (m == MISSING || v < m)
						m = v;
				}
			}
			return m;
		}

		// Returns the max value in the specified column
		public double ColumnMax(int col)
		{
			double m = MISSING;
			for (int i = 0; i < Rows(); i++)
			{
				double v = Get(i, col);
				if (v != MISSING)
				{
					if (m == MISSING || v > m)
						m = v;
				}
			}
			return m;
		}

		// Returns the original min value in the specified column
		public double ColumnMinOrig(int col)
		{
			return m_minOrig[col];
		}

		// Returns the max value in the specified column
		public double ColumnMaxOrig(int col)
		{
			return m_maxOrig[col];
		}

		// Returns the most common value in the specified column
		public double MostCommonValue(int col)
		{
			Dictionary<Double, int> tm = new Dictionary<Double, int>();
			for (int i = 0; i < Rows(); i++)
			{
				double v = Get(i, col);
				if (v != MISSING)
				{
					if (!tm.ContainsKey(v))
					{
						tm.Add(v, 1);
					}
					else
					{
						tm[v] += 1;
					}
				}
			}
			int maxCount = 0;
			double val = MISSING;
			foreach (var e in tm)
			{
				if (e.Value > maxCount)
				{
					maxCount = e.Value;
					val = e.Key;
				}
			}
			return val;
		}

		public void Normalize()
		{
			for (int i = 0; i < Cols(); i++)
			{
				if (ValueCount(i) == 0)
				{
					double min = ColumnMin(i);
					double max = ColumnMax(i);
					for (int j = 0; j < Rows(); j++)
					{
						double v = Get(j, i);
						if (v != MISSING)
						{
							if (max != min)
							{
								Set(j, i, (v - min) / (max - min));
							}
							else if (max != 0)
							{
								Set(j, i, 1);
							}
						}
					}
				}
			}
		}

		public void Print()
		{
			System.Console.WriteLine("@RELATION Untitled");
			for (int i = 0; i < m_attr_name.Count; i++)
			{
				System.Console.Write("@ATTRIBUTE " + m_attr_name[i]);
				int vals = ValueCount(i);
				if (vals == 0)
				{
					System.Console.WriteLine(" CONTINUOUS");
				}
				else
				{
					System.Console.Write(" {");
					for (int j = 0; j < vals; j++)
					{
						if (j > 0)
						{
							System.Console.Write(", ");
						}
						System.Console.Write(m_enum_to_str[i][j]);
					}
					System.Console.WriteLine("}");
				}
			}
			System.Console.WriteLine("@DATA");
			for (int i = 0; i < Rows(); i++)
			{
				double[] r = Row(i);
				for (int j = 0; j < r.Length; j++)
				{
					if (j > 0)
					{
						System.Console.Write(", ");
					}
					if (ValueCount(j) == 0)
					{
						System.Console.Write(r[j]);
					}
					else
					{
						System.Console.Write(m_enum_to_str[j][(int)r[j]]);
					}
				}
				System.Console.WriteLine("");
			}
		}
	}
}