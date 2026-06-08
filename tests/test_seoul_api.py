import unittest

from app.seoul_api import parse_population_xml


SAMPLE_XML = """
<Map>
  <SeoulRtd.citydata_ppltn>
    <AREA_NM>광화문·덕수궁</AREA_NM>
    <AREA_CD>POI009</AREA_CD>
    <AREA_CONGEST_LVL>여유</AREA_CONGEST_LVL>
    <AREA_CONGEST_MSG>사람이 몰려있을 가능성이 낮습니다.</AREA_CONGEST_MSG>
    <AREA_PPLTN_MIN>18000</AREA_PPLTN_MIN>
    <AREA_PPLTN_MAX>20000</AREA_PPLTN_MAX>
    <MALE_PPLTN_RATE>47.0</MALE_PPLTN_RATE>
    <FEMALE_PPLTN_RATE>53.0</FEMALE_PPLTN_RATE>
    <RESNT_PPLTN_RATE>21.5</RESNT_PPLTN_RATE>
    <NON_RESNT_PPLTN_RATE>78.5</NON_RESNT_PPLTN_RATE>
    <PPLTN_TIME>2026-06-07 17:50</PPLTN_TIME>
  </SeoulRtd.citydata_ppltn>
  <RESULT>
    <RESULT.CODE>INFO-000</RESULT.CODE>
    <RESULT.MESSAGE>정상 처리되었습니다.</RESULT.MESSAGE>
  </RESULT>
</Map>
"""


class SeoulApiParserTest(unittest.TestCase):
    def test_parse_population_xml_supports_real_map_shape(self):
        observation = parse_population_xml(SAMPLE_XML)

        self.assertEqual(observation.area_name, "광화문·덕수궁")
        self.assertEqual(observation.area_code, "POI009")
        self.assertEqual(observation.congestion_level, "여유")
        self.assertEqual(observation.population_min, 18000)
        self.assertEqual(observation.population_max, 20000)
        self.assertEqual(observation.population_midpoint, 19000)
        self.assertEqual(observation.observed_at.hour, 17)


if __name__ == "__main__":
    unittest.main()
