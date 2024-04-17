%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% File:     pwrdpl.cls           (LaTeX Document class option "pwrdpl")
%          
% Author:   Bartosz Błyszcz
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\message{*** [pwrdpl.cls] ---  (BB)  v1.0 <2024-02-29>  ***}

\newif\ifpdflatex\pdflatexfalse
\NeedsTeXFormat{LaTeX2e} % 
\ProvidesClass{pwrdpl}
\RequirePackage{mmap}
\RequirePackage{kvoptions}
\RequirePackage{csvsimple}
\RequirePackage{lscape}
\DeclareStringOption[pl]{language}
\ProcessKeyvalOptions*

\DeclareOption{pdflatex}{\pdflatextrue}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\LoadClass[a4paper,twoside,openany]{report}

\RequirePackage{xstring}

% \show\pwrdpl@language
\newcommand{\ThesisLanguage}{pl}
\IfStrEqCase{\pwrdpl@language}{%
	{en}{\renewcommand{\ThesisLanguage}{en}}%
	{pl}{\renewcommand{\ThesisLanguage}{pl}}%
}[\PackageError{pwrdpl}{Invalid thesis language.}{Available languages: Polish (pl), English (en)}]

% \show\ThesisLanguage

%\RequirePackage{fontspec}
\RequirePackage{tgtermes}
%%\setmainfont{Times New Roman}
%\setsansfont{TeX Gyre Heros}

% Lista wszystkich języków stanowiących języki pozycji bibliograficznych użytych w pracy.
% (Zgodnie z zasadami tworzenia bibliografii każda pozycja powinna zostać utworzona zgodnie z zasadami języka, w którym dana publikacja została napisana.)
\RequirePackage[english,polish]{babel}

% Użyj polskiego łamania wyrazów (zamiast domyślnego angielskiego).
\RequirePackage{polski}

% Użyj kodowania UTF-8.
\RequirePackage[utf8]{inputenc}

% Pisanie pseudokodu
\RequirePackage{algorithm,algpseudocode}

\RequirePackage{caption}

\RequirePackage{mathtools}
\RequirePackage{amsfonts}
\RequirePackage{amsmath}
\RequirePackage{amsthm}

\RequirePackage{graphicx}
\RequirePackage{rotating}
\RequirePackage[usenames]{color}
\RequirePackage{url}
\RequirePackage{titlesec}
\RequirePackage{tocloft}
\RequirePackage{float}

\RequirePackage{nicematrix}

\RequirePackage{pgfplots}
\RequirePackage{tikz}
\usetikzlibrary{shapes.symbols}
\usetikzlibrary{shapes.geometric}
\usetikzlibrary{shapes.arrows}

\RequirePackage[none]{hyphenat}
% \RequirePackage[all]{nowidow}
% \usepackage[hyphenation]{impnattypo}
\RequirePackage{diagbox}


\RequirePackage[font=small, labelfont=bf, labelsep=period]{caption}
\RequirePackage{subcaption}
\RequirePackage{enumerate}
\RequirePackage{enumitem}
\RequirePackage{colortbl}
\RequirePackage{hhline}

\newcolumntype{P}[1]{>{\centering\arraybackslash}p{#1}}
\newcolumntype{M}[1]{>{\centering\arraybackslash}m{#1}}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}m{#1}}

\setlist[enumerate]{itemsep=-2pt}
\setlist[itemize]{itemsep=-2pt}
% =============================================================================
% SEC:   Ustawienia związane z formatowaniem tekstu
% =============================================================================

\RequirePackage{geometry}

\geometry{outer=2.5cm, inner=3cm, tmargin=2.5cm, bmargin=2.5cm, foot=10mm, head=10mm}  % rozmiary marginesów i nagłówka/stopki
\linespread{1}  % interlinia
\renewcommand{\labelitemi}{--}  % znak wypunktowania użyty w liście wypunktowania

\RequirePackage[all,defaultlines=2]{nowidow} % Kontrola sierot i wdów - działa lepiej niż \widowpenalty i \clubpenalty
\brokenpenalty=10000\relax
\setlength\parindent{0pt}
% =============================================================================
% SEC:   Polecenia wymagane m.in. do generowania strony tytułowej
% =============================================================================

\def\titlePL#1        {\gdef\@titlePL{#1}}
\def\faculty#1      {\gdef\@faculty{#1}}
\def\thesistype#1   {\gdef\@thesistype{#1}}
\def\supervisor#1   {\gdef\@supervisor{#1}}
\def\acknowledgements#1   {\gdef\@acknowledgements{#1}}
\def\department#1   {\gdef\@department{#1}}
\def\degreeprogramme#1 {\gdef\@degreeprogramme{#1}}

%------------------------------------------------------------------------------
\newcommand{\titlepages}{%

	\titlepage

	\begin{center}
		{\Large{\bfseries{Politechnika Wrocławska}}}\linebreak
		\vspace*{1mm}
		{\large\bfseries \@faculty}
		\vspace*{1mm}

		\hrule

		\vspace*{3mm}
		\noindent {
			\begin{tabular}{l l}
			Kierunek: & \textbf{Informatyka Techniczna (ITE)} \\
			Specjalność: & \textbf{Inżynieria Systemów Informatycznych (INS)}
			\end{tabular}
		}

		\vspace*{15mm}
		{\LARGE\scshape{\@thesistype}}
		\vspace*{15mm}

		{\Large
			\IfStrEqCase{\ThesisLanguage}{%
				{pl}{\textbf{\@titlePL}}
				{en}{\textbf{\@titleEN}}
			}
		}

		\vspace*{8mm}
		{\large inż. Bartosz Błyszcz}\linebreak
		\vspace*{4mm}\linebreak

		{\large Opiekun pracy\\\vspace*{1mm}\textbf{dr inż. Tomasz Babczyński}}\linebreak

	\end{center}
	\vspace{\fill}
	\noindent{\small{Słowa kluczowe: 3-6 słów}}
	\vspace*{6mm}
	\hrule
	\vspace*{1mm}
	\centerline{\small WROCŁAW, \@date}

	\clearpage
	\thispagestyle{empty}

	\clearpage
	\thispagestyle{empty}

	~ \newpage

	% Obecna strona będzie mieć nr 5.
	\setcounter{page}{3}
}


% =============================================================================
% SEC:   Ustawienia nagłówków i stopek
% =============================================================================

\RequirePackage{fancyhdr}

\pagestyle{fancy}

\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0.25pt}
\fancyhf{}
\fancyhead{}
\fancyfoot{}
\fancyfoot[RO,LE]{\bfseries \thepage}

\fancypagestyle{plain}{%
	\fancyhf{}
	\fancyhead{}
	\fancyfoot{}
	\fancyfoot[RO,LE]{\bfseries \thepage}
}

\frenchspacing
\sloppy
\flushbottom


% =============================================================================
% SEC:   Ustawienia formatowania tytułów rozdziałów, podrozdziałów, itd.
% =============================================================================

\titleformat{\chapter}[block]
{\bfseries\LARGE}{\filright \LARGE\thechapter. }{0ex}{}
%odstępy: lewy, góra, dół
\titlespacing{\chapter}{0pt}{8mm}{8mm}
\titleformat{\section}[block]
{\bfseries\Large}{\filright \Large\thesection. }{0ex}{}
\titlespacing{\section}{7mm}{8mm plus 0mm minus 1mm}{4mm plus 0mm minus 1mm}
\titleformat{\subsection}[block]
{\bfseries\large}{\filright \large\thesubsection. }{0ex}{}
\titlespacing{\subsection}{7mm}{6mm}{2.5mm}
\titleformat{\subsubsection}[block]
{\bfseries\normalsize}{\filright \normalsize\thesubsubsection. }{0ex}{}
\titlespacing{\subsubsection}{7mm}{3mm}{1.5mm}

% Określ maksymalny poziom numeracji (głębsze poziomy nie będą miały numeru).
%   (np. 1 - chapter, 2 - section, itd.)
\setcounter{secnumdepth}{4}


% =============================================================================
% SEC:   Ustawienia generowania spisu treści i spisu rysunków
% =============================================================================

\renewcommand{\cfttoctitlefont}{\bfseries\Large}
\renewcommand{\cftbeforetoctitleskip}{20mm}
\renewcommand{\cftaftertoctitleskip}{19mm}
\renewcommand{\cftchapleader}{\cftdotfill{\cftdot}}
\renewcommand{\cftsecleader}{\cftdotfill{\cftdot}}
\renewcommand{\cftsubsecleader}{\cftdotfill{\cftdot}}
\renewcommand{\cftchapaftersnum}{.}
\renewcommand{\cftsecaftersnum}{.}
\renewcommand{\cftsubsecaftersnum}{.}
\setlength{\cftparskip}{2pt}
\setlength{\cftbeforechapskip}{0mm}
\renewcommand{\cftchappagefont}{\normalsize}


% =============================================================================
% SEC:   Ustawienia wyświetlania liczb (zgodne z polskimi zwyczajami typograficznymi)
% =============================================================================

\RequirePackage{siunitx}

\sisetup{
	output-decimal-marker = {,},  % symbol dziesiętny
	%	round-mode=places,
	%	round-precision=4,
	group-separator={~},  % separator grupy 3 cyfr
}

% =============================================================================
% SEC:   Ustawienia wyświetlania URL-i
% =============================================================================

% Zmiana kroju czcionki URL-i z czcionki o stałej szerokości na szeryfową z kursywą.
\renewcommand\UrlFont{\rmfamily\itshape}

% =============================================================================
% SEC:   Ustawienia podpisów pod ilustracjami
% =============================================================================

\captionsetup[subfigure]{labelfont=md}
\captionsetup{%
	% Użyj okrągłych nawiasów wokół odwołań do "podilustracji".
	subrefformat=parens,
	% Margines z obu stron podpisu.
	margin=2cm,
	% W przypadku podpisów mieszczących się w jednej linii nie stosuj justowania, tylko wyśrodkuj cały podpis.
	singlelinecheck=on,
}


% =============================================================================
%    SEC:   Konfiguracja bibliografii
% =============================================================================

\RequirePackage[
	% zastosowanie kompresji numerów czyli zamiast [1,2,3] jest [1-3]
	style=numeric-comp,
	sorting=none,
	%
	% Zastosuj styl wpisu bibliograficznego właściwy językowi publikacji.
	language=autobib,
	autolang=other,
	% Zapisuj datę dostępu do strony WWW w formacie RRRR-MM-DD.
	urldate=iso8601,
	% Nie dodawaj numerów stron, na których występuje cytowanie.
	backref=false,
	% Podawaj ISBN.
	isbn=true,
	% Nie podawaj URL-i, o ile nie jest to konieczne.
	url=false,
	%
	% Ustawienia związane z polskimi normami dla bibliografii.
	maxbibnames=3,
	% Wybór silnika generowania bibliografii:
	% backend=bibtex  % - jeżeli używamy BibTeX-a
	backend=biber   % - jeżeli używamy biber-a
]{biblatex}

\RequirePackage{csquotes}
% Ponieważ `csquotes` nie posiada polskiego stylu, można skorzystać z mocno zbliżonego stylu chorwackiego.
\DeclareQuoteAlias{croatian}{polish}

\addbibresource{bibliografia.bib}

% Nie wyświetlaj wybranych pól.
%\AtEveryBibitem{\clearfield{note}}

\DeclareFieldFormat*{title}{,,\textit{#1}''}


% =============================================================================
%    SEC:   Konfiguracja listingów
% =============================================================================

% Użyj czcionki kroju Courier.
\RequirePackage{courier}

\RequirePackage{listings}
\lstloadlanguages{TeX}

\lstset{
	literate={ą}{{\k{a}}}1
           {ć}{{\'c}}1
           {ę}{{\k{e}}}1
           {ó}{{\'o}}1
           {ń}{{\'n}}1
           {ł}{{\l{}}}1
           {ś}{{\'s}}1
           {ź}{{\'z}}1
           {ż}{{\.z}}1
           {Ą}{{\k{A}}}1
           {Ć}{{\'C}}1
           {Ę}{{\k{E}}}1
           {Ó}{{\'O}}1
           {Ń}{{\'N}}1
           {Ł}{{\L{}}}1
           {Ś}{{\'S}}1
           {Ź}{{\'Z}}1
           {Ż}{{\.Z}}1,
		   captionpos=b,
    % backgroundcolor=\color{lstBackground},
    commentstyle=\ttfamily\itshape\color{gray},
    tabsize = 4, %% set tab space width
    showstringspaces = false, %% prevent space marking in strings, string is defined as the text that is generally printed directly to the console
    numbers = left, %% display line numbers on the left
    keywordstyle = \color{blue}, %% set keyword color
    stringstyle = \color{red}, %% set string color
    rulecolor = \color{black}, %% set frame color to avoid being affected by text color
    basicstyle = \small \ttfamily , %% set listing font and size
    breaklines = true, %% enable line breaking
    postbreak=\mbox{\textcolor{red}{$\hookrightarrow$}\space},
    basicstyle=\ttfamily,
    columns=fullflexible,
    numberstyle = \tiny,
	basicstyle=\footnotesize\ttfamily,
}

\renewcommand{\lstlistingname}{Fragment kodu}
% =============================================================================
%    SEC:   Konfiguracja tabel
% =============================================================================

\RequirePackage{array}
\RequirePackage{tabularx}
\RequirePackage{multirow}
\RequirePackage{booktabs}
\RequirePackage{makecell}
\RequirePackage[flushleft]{threeparttable}

% defines the X column to use m (\parbox[c]) instead of p (`parbox[t]`)
\newcolumntype{C}[1]{>{\hsize=#1\hsize\centering\arraybackslash}X}

% =============================================================================
%    SEC:   INNE
% =============================================================================

\newcommand{\RedefinePlainStyle}{%
	% Ponowne zdefiniowanie stylu `plain`, aby usunąć numer strony z pierwszej strony spisu treści i poszczególnych rozdziałów.
	\fancypagestyle{plain}
	{
		% Usuń nagłówek i stopkę
		\fancyhf{}
		% Usuń linie.
		\renewcommand{\headrulewidth}{0pt}
		\fancyfoot{}
		\fancyfoot[RO,LE]{\bfseries \thepage}
	}
}

\AtBeginDocument{
	\renewcommand{\tablename}{Tabela}
	\renewcommand{\figurename}{Rys.}
}

\newcommand{\trans}[1]{(\textit{#1})}


% =============================================================================
% SEC:   grafy
% =============================================================================

\tikzstyle{startstop} = [rectangle, rounded corners, minimum width=3cm, minimum height=1cm,text centered, draw=black, fill=strongorange!50]
\tikzstyle{io} = [trapezium, trapezium left angle=70, trapezium right angle=110, minimum width=3cm, minimum height=1cm, text centered, draw=black, fill=blue!30]
\tikzstyle{process} = [rectangle, minimum width=3cm, minimum height=1cm, text centered, draw=black, fill=lighteryellow!30]
\tikzstyle{decision} = [diamond, minimum width=3cm, minimum height=1cm, text centered, draw=black, fill=lightblue!50, align=center]
\tikzstyle{arrow} = [thick,->,>=stealth]
\tikzstyle{backarrow} = [thick,<-,>=stealth]


% =============================================================================
% SEC:   Kolory
% =============================================================================

\definecolor{darkgreen}{RGB}{  0 104  56}
\definecolor{lightblue}{RGB}{181 212 239}
\definecolor{strongorange}{RGB}{247 148  30}
\definecolor{lighteryellow}{RGB}{242 245 205}
\definecolor{lightred}{RGB}{252 134 134}
\definecolor{lightgreen}{RGB}{134 252 167}
% =============================================================================
% SEC:   Niestandardowe środowiska
% =============================================================================

% ----------------------------------------------------------------------------
%    Środowisko do tworzenia opisów symboli użytych w równaniu
% ----------------------------------------------------------------------------

\SetLabelAlign{parright}{\parbox[t]{\labelwidth}{\raggedleft#1}}
\setlist[description]{style=multiline, topsep=10pt, align=parright}

% Odstęp po obu stronach półpauzy.
\newlength{\descriptiondashgap}
\setlength{\descriptiondashgap}{0.5em}

% Element na liście symboli matematycznych użytych w równaniu (symbol wyrównany do prawej, oddzielony półpauzą od definicji)
\renewcommand*\descriptionlabel[1]{\hspace \labelsep \normalfont \bfseries #1 \mdseries\hspace{\descriptiondashgap}--\hspace{\descriptiondashgap}}

% Odległość półpauzy od lewego marginesu.
\newlength{\eqwheremargin}
\setlength{\eqwheremargin}{2cm}

\newenvironment{eqwhere}[1][]
{%
	\setlength{\eqwheremargin}{#1}%
	\begin{description}[leftmargin=\eqwheremargin, itemsep=0cm, labelsep=0cm]
}
{\end{description}}



% =============================================================================
% SEC:   Kapitaliki
% =============================================================================

% ----------------------------------------------------------------------------
%    Konfiguracja kapitalików dla Times New Roman
% ----------------------------------------------------------------------------

%\newfontfamily\tgtermes{TeX Gyre Termes}
%\makeatletter
%  \begingroup
%    \tgtermes
%    \DeclareFontShape{\f@encoding}{\rmdefault}{m}{sc}{%
%      <-> ssub * \f@family/m/sc}{}
%    \DeclareFontShape{\f@encoding}{\rmdefault}{bx}{sc}{%
%      <-> ssub * \f@family/bx/sc}{}
%      \DeclareFontShape{\f@encoding}{\rmdefault}{b}{sc}{%
%      <-> ssub * \f@family/b/sc}{}
%  \endgroup
%\makeatother



% =============================================================================
% SEC:   Caption ze źródłem
% =============================================================================

\newcommand*{\captionsource}[2]{%
	\caption{#1}
	{
		\vspace{-10pt}
		\caption*{Źródło: #2}
	}
}

\newcommand*{\captionsourceb}[3]{%
\caption{#1}
{
\vspace{-10pt}
\caption*{#2}
\vspace{-10pt}
\caption*{Źródło: #3}
}
}

\newcommand*{\refsource}[2]{%
\textbf{#1}\;
\textbf{\ref{#2}}}
