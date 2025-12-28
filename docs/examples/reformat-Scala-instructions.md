
- Add types for local variables and method returns. Do not make assumptions about the type - if type cannot be 
inferred from the provided code alone, omit type definition.
  
- Break long statements up in several lines, use intermediate variables
  
- When dealing with Java code, handle null values explicitly and as soon as possible
  - Don't add `null` checks unless they are really needed
  - Don't wrap `null`-s to `Option` unless you pass it further to Scala code
  - Handle the `null`-s immediately if possible, just wrapping them to `Option` pushes the responsibility to the receiver of the `Option`.
  
- Get rid of `Option`-s as early as possible, prefer pattern matching over call chaining
  
- Don't use infix or postfix notation, use dot notation with parenthesis everywhere: `obj.method(args)`
  
- Chain method calls with dots on new line
  
- Always use braces `()` in method definition and calls

- Use curly braces `{}` in method definitions
  
- Prefer code readability over complex pure functional code
  
- Prefer for comprehension over chaining async method calls
  
- Don't use curly braces for if-else with a single statement:
  ```scala
  if (playerContribution <= 2)
    1 + Math.floor(titanStars / 2.0).toInt
  else
    1
  ```
  
- Don't use curly braces for if with return:
  ```scala
  if (playerContribution <= 2) return 1 + Math.floor(titanStars / 2.0).toInt
  ```
