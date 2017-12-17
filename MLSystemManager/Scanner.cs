using System.IO;
using System.Text;

namespace MLSystemManager
{
	public class Scanner : StringReader
	{
		private bool _fetchNext = true;
		private string _nextWord = string.Empty;
		private char _delimiter = char.MinValue;

		public Scanner(string source) : base(source)
		{
		}

		public string Next()
		{
			if (_fetchNext)
			{
				GetNextWord();
			}

			_fetchNext = !string.IsNullOrEmpty(_nextWord);

			return _nextWord;
		}

		public void UseDelimiter(char delimiter)
		{
			_delimiter = delimiter;
		}

		public bool HasNext()
		{
			if (_fetchNext)
			{
				GetNextWord();
			}

			_fetchNext = false;

			return _nextWord.Length > 0;
		}

		private void GetNextWord()
		{
			System.Text.StringBuilder sb = new StringBuilder();
			char nextChar;
			int next;
			bool skipWhiteSpace = true;
			do
			{
				next = this.Read();
				if (next < 0)
				{
					break;
				}
				nextChar = (char)next;
				if (char.IsWhiteSpace(nextChar))
				{
					if (skipWhiteSpace)
					{
						continue;
					}
					else if (_delimiter == char.MinValue)
					{
						break;
					}
				}
				else if (nextChar == _delimiter)
				{
					break;
				}
				sb.Append(nextChar);
				skipWhiteSpace = false;
			} while (true);
			while ((this.Peek() >= 0) && (char.IsWhiteSpace((char)this.Peek()) || ((char)this.Peek() == _delimiter)))
			{
				this.Read();
			}

			_nextWord = sb.ToString();
		}
	}
}