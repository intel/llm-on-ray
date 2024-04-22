import string


example_scala_code = """
@Description(
        name = "norm_str",
        value = "_FUNC_(input, [defaultValue], [dirtyValues ...]) trims input and " +
                "normalize null, empty or dirtyValues to defVal. \n",
        extended = "preset defaultValue is 'N-A' and preset dirtyValues are {'null', 'unknown', 'unknow', 'N-A'},\n" +
                   "the third NULL argument will clear the preset dirtyValues list."
)
public class UDFNormalizeString extends GenericUDF {


    public final static String DEFAULT_VALUE = "N-A";

    @SuppressWarnings("SpellCheckingInspection")
    public final static List<String> DEFAULT_NULL_VALUES = Arrays.asList("null", "unknown", "unknow", DEFAULT_VALUE);

    private transient String defaultValue;
    private transient Set<String> nullValues;

    @Override
    public ObjectInspector initialize(ObjectInspector[] arguments) throws UDFArgumentException {

        if (arguments.length == 0) {
            throw new UDFArgumentLengthException("norm_str() expects at least one argument.");
        }

        defaultValue = DEFAULT_VALUE;
        if (arguments.length >= 2) {

            // ............
            if (!ObjectInspectorUtils.isConstantObjectInspector(arguments[1])) {
                throw new UDFArgumentTypeException(1, "norm_str() expects a constant value as default.");
            }

            // .....
            Object writable = ObjectInspectorUtils.getWritableConstantValue(arguments[1]);
            defaultValue = (writable == null ? null : writable.toString());
        }

        nullValues = new HashSet<>(DEFAULT_NULL_VALUES);
        for (int i = 2; i < arguments.length; i++) {

            if (!ObjectInspectorUtils.isConstantObjectInspector(arguments[i])) {
                throw new UDFArgumentTypeException(i, "norm_str() expects constant values as dirty values");
            }

            Object writable = ObjectInspectorUtils.getWritableConstantValue(arguments[i]);

            if (writable == null) {
                // .........null .......
                if (i != 2) {
                    throw new UDFArgumentException(
                            "Only the third null argument will clear the default null values of norm_str().");
                }
                nullValues.clear();
            } else {
                nullValues.add(writable.toString().trim().toLowerCase());
            }
        }

        return PrimitiveObjectInspectorFactory.javaStringObjectInspector;
    }

    @Override
    public Object evaluate(DeferredObject[] arguments) throws HiveException {
        assert arguments.length > 0;

        Object inputObject = arguments[0].get();

        if (inputObject == null) {
            return defaultValue;
        }

        String input = inputObject.toString().trim();

        if (input.length() == 0 || nullValues.contains(input.toLowerCase())) {
            return defaultValue;
        }

        return input;
    }

    @Override
    public String getDisplayString(String[] children) {
        return getStandardDisplayString("norm_str", children);
    }
}

"""

demo_sample_code = """
@Description(
        name = "norm_str",
        value = "_FUNC_(input, [defaultValue], [dirtyValues ...]) trims input and " +
                "normalize null, empty or dirtyValues to defVal. \n",
        extended = "preset defaultValue is 'N-A' and preset dirtyValues are {'null', 'unknown', 'unknow', 'N-A'},\n" +
                   "the third NULL argument will clear the preset dirtyValues list."
)
public class UDFNormalizeString extends GenericUDF {


    public final static String DEFAULT_VALUE = "N-A";

    @SuppressWarnings("SpellCheckingInspection")
    public final static List<String> DEFAULT_NULL_VALUES = Arrays.asList("null", "unknown", "unknow", DEFAULT_VALUE);

    private transient String defaultValue;
    private transient Set<String> nullValues;

    @Override
    public ObjectInspector initialize(ObjectInspector[] arguments) throws UDFArgumentException {

        if (arguments.length == 0) {
            throw new UDFArgumentLengthException("norm_str() expects at least one argument.");
        }

        defaultValue = DEFAULT_VALUE;
        if (arguments.length >= 2) {

            // ............
            if (!ObjectInspectorUtils.isConstantObjectInspector(arguments[1])) {
                throw new UDFArgumentTypeException(1, "norm_str() expects a constant value as default.");
            }

            // .....
            Object writable = ObjectInspectorUtils.getWritableConstantValue(arguments[1]);
            defaultValue = (writable == null ? null : writable.toString());
        }

        nullValues = new HashSet<>(DEFAULT_NULL_VALUES);
        for (int i = 2; i < arguments.length; i++) {

            if (!ObjectInspectorUtils.isConstantObjectInspector(arguments[i])) {
                throw new UDFArgumentTypeException(i, "norm_str() expects constant values as dirty values");
            }

            Object writable = ObjectInspectorUtils.getWritableConstantValue(arguments[i]);

            if (writable == null) {
                // .........null .......
                if (i != 2) {
                    throw new UDFArgumentException(
                            "Only the third null argument will clear the default null values of norm_str().");
                }
                nullValues.clear();
            } else {
                nullValues.add(writable.toString().trim().toLowerCase());
            }
        }

        return PrimitiveObjectInspectorFactory.javaStringObjectInspector;
    }

    @Override
    public Object evaluate(DeferredObject[] arguments) throws HiveException {
        assert arguments.length > 0;

        Object inputObject = arguments[0].get();

        if (inputObject == null) {
            return defaultValue;
        }

        String input = inputObject.toString().trim();

        if (input.length() == 0 || nullValues.contains(input.toLowerCase())) {
            return defaultValue;
        }

        return input;
    }

    @Override
    public String getDisplayString(String[] children) {
        return getStandardDisplayString("norm_str", children);
    }
}
"""


convert_to_cpp_temp = """

Convert the following code into a C++function or class:
```
{}
```
"""

example_temp = string.Template(
    """
Your task is to refer example Velox UDF and rewrite code I provided into a Velox UDF.
Following code is an example Velox UDF:
```
#include <velox/expression/VectorFunction.h>
#include <iostream>
#include "udf/Udf.h"

namespace {
using namespace facebook::velox;

template <TypeKind Kind>
class PlusConstantFunction : public exec::VectorFunction {
 public:
  explicit PlusConstantFunction(int32_t addition) : addition_(addition) {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    using nativeType = typename TypeTraits<Kind>::NativeType;
    VELOX_CHECK_EQ(args.size(), 1);

    auto& arg = args[0];

    // The argument may be flat or constant.
    VELOX_CHECK(arg->isFlatEncoding() || arg->isConstantEncoding());

    BaseVector::ensureWritable(rows, createScalarType<Kind>(), context.pool(), result);

    auto* flatResult = result->asFlatVector<nativeType>();
    auto* rawResult = flatResult->mutableRawValues();

    flatResult->clearNulls(rows);

    if (arg->isConstantEncoding()) {
      auto value = arg->as<ConstantVector<nativeType>>()->valueAt(0);
      rows.applyToSelected([&](auto row) { rawResult[row] = value + addition_; });
    } else {
      auto* rawInput = arg->as<FlatVector<nativeType>>()->rawValues();

      rows.applyToSelected([&](auto row) { rawResult[row] = rawInput[row] + addition_; });
    }
  }

 private:
  const int32_t addition_;
};

static std::vector<std::shared_ptr<exec::FunctionSignature>> integerSignatures() {
  // integer -> integer
  return {exec::FunctionSignatureBuilder().returnType("integer").argumentType("integer").build()};
}

static std::vector<std::shared_ptr<exec::FunctionSignature>> bigintSignatures() {
  // bigint -> bigint
  return {exec::FunctionSignatureBuilder().returnType("bigint").argumentType("bigint").build()};
}

} // namespace

const int kNumMyUdf = 2;
gluten::UdfEntry myUdf[kNumMyUdf] = {{"myudf1", "integer"}, {"myudf2", "bigint"}};

DEFINE_GET_NUM_UDF {
  return kNumMyUdf;
}

DEFINE_GET_UDF_ENTRIES {
  for (auto i = 0; i < kNumMyUdf; ++i) {
    udfEntries[i] = myUdf[i];
  }
}

DEFINE_REGISTER_UDF {
  facebook::velox::exec::registerVectorFunction(
      "myudf1", integerSignatures(), std::make_unique<PlusConstantFunction<facebook::velox::TypeKind::INTEGER>>(5));
  facebook::velox::exec::registerVectorFunction(
      "myudf2", bigintSignatures(), std::make_unique<PlusConstantFunction<facebook::velox::TypeKind::BIGINT>>(5));
  LOG(INFO) << "registered myudf1, myudf2";
}

```

$reference

Think step by step:
1. Understand the code I provided and the Velox UDF examples :
2. Consider whether there are any related classes or functions already implemented in velox in the code that needs to be rewritten, such as {$queries}
3. Rewrite the code as a Velox UDF

Please convert blow code:
```
$cpp_code
```
"""
)

example_related_queries = (
    "velox string functions, velox string normalization, and velox string case conversion"
)

generate_search_query_prompt = string.Template(
    """
Your task is to rewrite the code I provided as a new function in the velox project. Based on the code I provided,  write 3 the necessary search keywords to gather information from the Velox code or document.
For example, you found that the code requires calculating the distance between two strings, so you need to find out if there are any functions in Velox that handle string types, such as hamming_distance, that can be directly called

## Rule
- Don't ask questions you already know and velox udf specification
- The main purpose is to find functions, type definitions, etc. that already exist and can be directly used in Velox

Here is code:
```
$cpp_code
```

Only respond in the following JSON format:
```
{
"Queries":[
<QUERY 1>,
"<QUERY 2>"
]
}
"""
)

rag_suffix = """
Some Velox code for reference:
```
{}
```
"""
