// MathJax 3 reads these settings before the script in mkdocs.yml loads.
// pymdownx.arithmatex (generic) emits \(...\) and \[...\] inside spans and divs
// with the class "arithmatex", so only those elements are typeset.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};
